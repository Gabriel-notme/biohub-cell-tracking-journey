from pathlib import Path
import os,json,time,argparse
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('POLARS_MAX_THREADS','2');os.environ['BIOHUB_BASE_REPO']='/workspace/biohub/baseline_validation/tracking_repo'
import numpy as np,torch
from scipy.optimize import linear_sum_assignment
from prepare_events import graph
from refine_events import structure,EventRefiner
from joint_refine import JointRefiner,JOINT_DEFAULT
from candidate_rows import candidate_rows
from visual_motion_features import visual_fork_features
R=Path('/workspace/biohub');D=R/'visual_motion_data';TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')

def select(name,group):
    folder='training_baseline_graphs' if group=='train' else 'prior_graphs_B2';data=json.loads((R/folder/(name+'.json')).read_text());nodes={int(k):v for k,v in data['nodes'].items()};edges=data['edges'];out,prev,frames,pos=structure(nodes,edges)
    gn,go,gp,gf,gpos=graph(TRAIN/(name+'.geff'));matched={}
    for t,gs in gf.items():
        ns=frames.get(t,[])
        if not ns:continue
        dist=np.linalg.norm(np.asarray([gpos[g] for g in gs])[:,None]-np.asarray([pos[n] for n in ns])[None],axis=-1);ii,jj=linear_sum_assignment(dist)
        for i,j in zip(ii,jj):
            if dist[i,j]<=6:matched[ns[j]]=gs[i]
    er,fr=candidate_rows(nodes,edges,JOINT_DEFAULT);rng=np.random.default_rng(int.from_bytes(name.encode()[-5:],'little')%2**32);selected={}
    for kind,candidates in [('edge',er),('fork',fr)]:
        rr=[];yy=[]
        for row in candidates:
            source,*targets=map(int,row)
            if source not in matched or matched[source] not in go:continue
            s=matched[source];ds=[];ambiguous=False
            for target in targets:
                if target in matched:ds.append(matched[target])
                elif min(np.linalg.norm(pos[target]-gpos[g]) for g in go[s])<=7:ambiguous=True;break
                else:ds.append(-1)
            if ambiguous:continue
            y=int(ds[0] in go[s]) if kind=='edge' else int(set(ds)==set(go[s]) and len(go[s])==2);rr.append(row);yy.append(y)
        yy=np.asarray(yy,np.float32);rr=np.asarray(rr,np.int64).reshape(-1,2 if kind=='edge' else 3);positive=np.flatnonzero(yy);negative=np.flatnonzero(yy==0);cap=6000 if kind=='edge' else 3000
        if len(negative)>cap:negative=rng.choice(negative,cap,replace=False)
        ix=np.r_[positive,negative];selected[kind]=(rr[ix],yy[ix])
    return nodes,selected

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args();D.mkdir(exist_ok=True);split=json.loads((R/'events/split.json').read_text());(D/'split.json').write_text(json.dumps(split));torch.set_num_threads(2)
    encoder=EventRefiner([R/'B1/best.pt',R/'B2_pretrained/best.pt'],TRAIN,cache_dir=R/'visual_motion_training_cache')
    jobs=[(n,g) for g in ['calibration','train'] for n in split[g]][a.shard::2]
    for name,group in jobs:
        if (D/(name+'.npz')).exists():continue
        started=time.time();nodes,selected=select(name,group);rows,y=selected['fork']
        with np.load(R/'motion_data'/(name+'.npz')) as d:
            assert np.array_equal(y,d['fork_y']);motion=d['fork_x']
        if len(rows):
            wanted=set(map(int,rows.ravel()));subset={n:v for n,v in nodes.items() if n in wanted};ids,banks=encoder.embeddings(name,subset);x=visual_fork_features(motion,rows,encoder.models,banks,{n:i for i,n in enumerate(ids)})
        else:x=np.empty((0,0),np.float32)
        np.savez_compressed(D/(name+'.npz'),fork_x=x,fork_y=y);print('VISUAL_MOTION_PREPARED',name,len(y),int(y.sum()),round(time.time()-started),flush=True)
    (D/('shard_'+str(a.shard)+'_ready.json')).write_text(json.dumps({'ready':True}))
    if all((D/('shard_'+str(i)+'_ready.json')).exists() for i in range(2)):(D/'ready.json').write_text(json.dumps({'ready':True}))
