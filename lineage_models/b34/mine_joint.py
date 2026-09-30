from pathlib import Path
import os,sys,json,time,argparse,shutil
os.environ.setdefault('POLARS_MAX_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','3')
import numpy as np,torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from scipy.spatial import cKDTree
from cell_event import Movie,SCALE,chain,edge_geometry,fork_geometry
from prepare_events import graph
R=Path('/workspace/biohub');TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train');REPO=R/'baseline_validation/tracking_repo'
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from predict_unet_transformer import load_model

@torch.inference_mode()
def detections(model,movie):
    found={};T=movie.shape[0]
    for start in range(0,T-1,12):
        ts=list(range(start,min(T-1,start+12)))
        batch=np.stack([np.stack([movie.frame(t)[::1,::2,::2],movie.frame(t+1)[::1,::2,::2]]) for t in ts]);batch=np.maximum(0,(batch-movie.low)/(movie.high-movie.low+1e-6))
        with sdpa_kernel(SDPBackend.MATH),torch.autocast('cuda',dtype=torch.float16):_,logits=model.encode(torch.from_numpy(batch).cuda())
        prob=logits[1].float().sigmoid();peaks=(prob==F.max_pool3d(prob,3,1,1))&(prob>.965)
        for i,t in enumerate(ts):found[t+1]=torch.nonzero(peaks[i,0]).cpu().numpy().astype(np.float32)*np.array([1,4,4],np.float32)
    return found

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2);a=p.parse_args();torch.set_num_threads(3);torch.backends.cudnn.benchmark=True
    source=R/'joint_data';dest=R/'joint_mined';dest.mkdir(exist_ok=True);cache=R/'training_detector_candidates';cache.mkdir(exist_ok=True)
    split=json.loads((source/'split.json').read_text());(dest/'split.json').write_text(json.dumps(split,indent=2))
    model,_,ds=load_model(REPO/'weights/unet_transformer/split_0/edge_predictor_best.pth',torch.device('cuda'));assert tuple(ds)==(1,4,4);model.eval()
    todo=[(g,n) for g in ['calibration','train'] for n in split[g]][a.shard::a.shards];reports=[];started=time.time()
    for group,name in todo:
        marker=dest/(name+'_mined.json')
        if marker.exists():reports.append(json.loads(marker.read_text()));continue
        with np.load(source/(name+'.npz')) as d:data={k:d[k] for k in d.files}
        nodes,out,prev,frames,pos=graph(TRAIN/(name+'.geff'));movie=Movie(TRAIN/(name+'.zarr'))
        cp=cache/(name+'.npz')
        if cp.exists():
            with np.load(cp) as d:det={int(k):d[k] for k in d.files}
        else:det=detections(model,movie);np.savez_compressed(cp,**{str(t):d for t,d in det.items()})
        trees={t:cKDTree(c*SCALE) for t,c in det.items() if len(c)}
        def future(t,j):
            points=[det[t][j]*SCALE]
            for dt in range(1,4):
                if t+dt not in trees:break
                distance,k=trees[t+dt].query(points[-1])
                if distance>8:break
                points.append(det[t+dt][k]*SCALE)
            return np.asarray(points,np.float32)
        er=[];eg=[];erel=[];fr=[];fg=[];frel=[]
        for i,s in enumerate(data['source_ids']):
            s=int(s);t=int(nodes[s]['t']);children=[d for d in out.get(s,[]) if nodes[d]['t']==t+1]
            if not children or t+1 not in trees:continue
            near=trees[t+1].query_ball_point(pos[s],16)
            near=[j for j in near if min(np.linalg.norm(det[t+1][j]*SCALE-pos[c]) for c in children)>7.0]
            near.sort(key=lambda j:np.linalg.norm(det[t+1][j]*SCALE-pos[s]));h=chain(s,prev,pos)
            for j in near[:4]:
                ff=future(t+1,j);delta=ff[0]-pos[s]
                er.append([i,0]);eg.append(edge_geometry(h,ff));erel.append([[0,0,0],delta.tolist()])
                if len(ff)>=2:
                    c=children[0];fr.append([i,0]);fg.append(fork_geometry(h,chain(c,out,pos),ff));frel.append([[0,0,0],(pos[c]-pos[s]).tolist(),delta.tolist()])
        for kind,rows,geom,relative in [('edge',er,eg,erel),('fork',fr,fg,frel)]:
            n=2 if kind=='edge' else 3;gd=16 if kind=='edge' else 28
            data[kind+'_rows']=np.r_[data[kind+'_rows'],np.asarray(rows,np.float32).reshape(-1,2)]
            data[kind+'_geom']=np.r_[data[kind+'_geom'],np.asarray(geom,np.float32).reshape(-1,gd)]
            data[kind+'_rel']=np.r_[data[kind+'_rel'],np.asarray(relative,np.float32).reshape(-1,n,3)]
        np.savez_compressed(dest/(name+'.npz'),**data)
        bank=dest/(name+'_patches.npy')
        if not bank.exists():bank.symlink_to(source/(name+'_patches.npy'))
        report={'movie':name,'group':group,'new_edge_negatives':len(er),'new_fork_negatives':len(fr),'elapsed':round(time.time()-started)};marker.write_text(json.dumps(report));reports.append(report)
        (dest/('mining_report_'+str(a.shard)+'.json')).write_text(json.dumps(reports,indent=2));print('JOINT_MINED',json.dumps(report),flush=True)
    (dest/('shard_'+str(a.shard)+'_ready.json')).write_text(json.dumps({'movies':len(reports)}))
    if all((dest/('shard_'+str(i)+'_ready.json')).exists() for i in range(a.shards)):(dest/'ready.json').write_text(json.dumps({'ready':True,'shards':a.shards,'context':7}))
    print('MINING_SHARD_COMPLETE',a.shard,flush=True)
