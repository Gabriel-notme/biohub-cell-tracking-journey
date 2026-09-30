"""Mine real detector distractors only on fitting movies; never label hidden data."""
from pathlib import Path
import sys,os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','2')
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from scipy.spatial import cKDTree
from cell_event import Movie,SCALE,chain,edge_geometry,fork_geometry
from prepare_events import graph
ROOT=Path('/workspace/biohub');DATA=ROOT/'events';TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')
REPO=ROOT/'baseline_setup/tracking_repo';sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from predict_unet_transformer import load_model

@torch.inference_mode()
def detections(model,movie):
    found={};T=movie.shape[0]
    for start in range(0,T-1,8):
        ts=list(range(start,min(T-1,start+8)))
        batch=np.stack([np.stack([movie.frame(t)[::1,::2,::2],movie.frame(t+1)[::1,::2,::2]]) for t in ts])
        batch=np.maximum(0,(batch-movie.low)/(movie.high-movie.low+1e-6))
        with sdpa_kernel(SDPBackend.MATH),torch.autocast('cuda',dtype=torch.bfloat16):
            _,logits=model.encode(torch.from_numpy(batch).cuda())
        prob=logits[1].float().sigmoid();peaks=(prob==F.max_pool3d(prob,3,1,1))&(prob>.98)
        for i,t in enumerate(ts):
            c=torch.nonzero(peaks[i,0]).cpu().numpy().astype(np.float32)*np.array([1,4,4],np.float32)
            found[t+1]=c
    return found

def run(data=None,context=5,cache=None):
    global DATA
    if data is not None:DATA=Path(data)
    torch.set_num_threads(3);torch.backends.cudnn.benchmark=True
    model,_,ds=load_model(REPO/'weights/unet_transformer/split_0/edge_predictor_best.pth',torch.device('cuda'))
    assert tuple(ds)==(1,4,4);model.eval()
    split=json.loads((DATA/'split.json').read_text());reports=[];started=time.time()
    for name in split['calibration']+split['train']:
        marker=DATA/(name+'_mined.json')
        if marker.exists():reports.append(json.loads(marker.read_text()));continue
        nodes,out,prev,frames,pos=graph(TRAIN/(name+'.geff'));movie=Movie(TRAIN/(name+'.zarr'),context=context)
        cache_path=Path(cache)/(name+'.npz') if cache else None
        if cache_path is not None and cache_path.exists():
            with np.load(cache_path) as data:det={int(k):data[k] for k in data.files}
        else:
            det=detections(model,movie)
            if cache_path is not None:np.savez_compressed(cache_path,**{str(k):v for k,v in det.items()})
        trees={t:cKDTree(c*SCALE) for t,c in det.items() if len(c)}
        old=np.load(DATA/(name+'.npz'));oldids=old['node_ids'].tolist();lookup={n:i for i,n in enumerate(oldids)}
        newcoords=[];newkeys={};er=[];eg=[];fr=[];fg=[]
        def future(t,index):
            point=det[t][index]*SCALE;points=[point]
            for dt in range(1,4):
                tt=t+dt
                if tt not in trees:break
                dist,j=trees[tt].query(points[-1])
                if dist>7.5:break
                points.append(det[tt][j]*SCALE)
            return np.asarray(points,np.float32)
        def bank_index(t,j):
            key=(t,int(j))
            if key not in newkeys:
                newkeys[key]=len(oldids)+len(newcoords);newcoords.append((t,det[t][j]))
            return newkeys[key]
        rng=np.random.default_rng(20260921)
        # Use exactly the fitting sources already sampled, and preserve annotated labels.
        chosen={oldids[int(i)] for i in old['edges'][:,0]}
        for s in sorted(chosen):
            t=int(nodes[s]['t']);children=out.get(s,[])
            if not children or t+1 not in trees:continue
            children=[c for c in children if nodes[c]['t']==t+1]
            if not children:continue
            near=trees[t+1].query_ball_point(pos[s],11.)
            near=[j for j in near if min(np.linalg.norm(det[t+1][j]*SCALE-pos[c]) for c in children)>4.5]
            near.sort(key=lambda j:np.linalg.norm(det[t+1][j]*SCALE-pos[s]))
            for j in near[:2]:
                ix=bank_index(t+1,j);ff=future(t+1,j);h=chain(s,prev,pos)
                er.append((lookup[s],ix,0));eg.append(edge_geometry(h,ff))
                a=children[0]
                if a in lookup and len(ff)>=2 and rng.random()<.3:
                    fr.append((lookup[s],lookup[a],ix,0));fg.append(fork_geometry(h,chain(a,out,pos),ff))
        if newcoords:
            oldbank=np.load(DATA/(name+'_patches.npy'),mmap_mode='r')
            newpath=DATA/(name+'_patches_mined.npy')
            bank=np.lib.format.open_memmap(newpath,mode='w+',dtype=np.float16,shape=(len(oldids)+len(newcoords),context,12,24,24))
            bank[:len(oldids)]=oldbank[:len(oldids)]
            for i in sorted(range(len(newcoords)),key=lambda i:newcoords[i][0]):
                t,c=newcoords[i];bank[len(oldids)+i]=movie.patch(t,c)
            bank.flush();del bank,oldbank;newpath.replace(DATA/(name+'_patches.npy'))
        oldphase=old['phase'].copy()
        for i,n in enumerate(oldids):
            if n not in out:oldphase[i]=-1
        phase=np.r_[oldphase,np.full(len(newcoords),-1,np.float32)]
        node_ids=np.r_[old['node_ids'],np.arange(len(newcoords))+max(nodes)+1000000]
        edges=np.r_[old['edges'],np.asarray(er,np.int32).reshape(-1,3)]
        edge_geom=np.r_[old['edge_geom'],np.asarray(eg,np.float32).reshape(-1,16)]
        forks=np.r_[old['forks'],np.asarray(fr,np.int32).reshape(-1,4)]
        fork_geom=np.r_[old['fork_geom'],np.asarray(fg,np.float32).reshape(-1,28)]
        old.close()
        np.savez_compressed(DATA/(name+'.npz'),edges=edges,edge_geom=edge_geom,forks=forks,fork_geom=fork_geom,phase=phase,node_ids=node_ids,split='train' if name in split['train'] else 'calibration')
        report={'movie':name,'new_detector_cells':len(newcoords),'new_edge_negatives':len(er),'new_fork_negatives':len(fr),'elapsed':round(time.time()-started)}
        marker.write_text(json.dumps(report));reports.append(report);print('MINED',json.dumps(report),flush=True)
        (DATA/'mining_report.json').write_text(json.dumps(reports,indent=2))
    (DATA/'mining_done.json').write_text(json.dumps({'movies':len(reports),'seconds':time.time()-started}))
    print('MINING_COMPLETE',flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--data');parser.add_argument('--context',type=int,default=5);parser.add_argument('--cache')
    args=parser.parse_args();run(args.data,args.context,args.cache)
