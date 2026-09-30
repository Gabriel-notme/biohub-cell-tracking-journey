from pathlib import Path
import os,json,time,argparse,multiprocessing as mp
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('POLARS_MAX_THREADS','2')
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from prepare_events import graph
from refine_events import structure
from joint_refine import JointRefiner,JOINT_DEFAULT
from motion_features import MotionFeatures,VERSION
R=Path('/workspace/biohub');TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train');DEST=R/'motion_data'

def prepare(name,group):
    started=time.time();path=DEST/(name+'.npz');marker=DEST/(name+'_ready.json')
    if marker.exists() and json.loads(marker.read_text()).get('label_version')==2:return json.loads(marker.read_text())
    folder='training_baseline_graphs' if group=='train' else 'prior_graphs_B2';d=json.loads((R/folder/(name+'.json')).read_text());nodes={int(k):v for k,v in d['nodes'].items()};edges=d['edges'];out,prev,frames,pos=structure(nodes,edges)
    gn,go,gp,gf,gpos=graph(TRAIN/(name+'.geff'));matched={}
    for t,gs in gf.items():
        ns=frames.get(t,[])
        if not ns:continue
        pp=np.asarray([pos[n] for n in ns]);gg=np.asarray([gpos[g] for g in gs]);distance=np.linalg.norm(gg[:,None]-pp[None],axis=-1)
        ii,jj=linear_sum_assignment(distance)
        for i,j in zip(ii,jj):
            if distance[i,j]<=6:matched[ns[j]]=gs[i]
    extractor=JointRefiner.__new__(JointRefiner);extractor.config=dict(JOINT_DEFAULT)
    er,fr,_,_=extractor.candidates(nodes,edges);rng=np.random.default_rng(int.from_bytes(name.encode()[-5:],'little')%2**32);selected={}
    for kind,candidates in [('edge',er),('fork',fr)]:
        rr=[];yy=[]
        for row in candidates:
            source,*targets=map(int,row)
            if source not in matched:continue
            s=matched[source]
            if s not in go:continue
            # A labeled mother's observed successors define its transition.
            # Unannotated neighboring cells are valid negatives only when far
            # from every annotated successor; close ambiguous peaks are ignored.
            ds=[];ambiguous=False
            for target in targets:
                if target in matched:ds.append(matched[target])
                else:
                    if min(np.linalg.norm(pos[target]-gpos[g]) for g in go[s])<=7:ambiguous=True;break
                    ds.append(-1)
            if ambiguous:continue
            y=int(ds[0] in go[s]) if kind=='edge' else int(set(ds)==set(go[s]) and len(go[s])==2)
            rr.append(row);yy.append(y)
        yy=np.asarray(yy,np.float32);rr=np.asarray(rr,np.int64).reshape(-1,2 if kind=='edge' else 3)
        posix=np.flatnonzero(yy);negix=np.flatnonzero(yy==0);cap=6000 if kind=='edge' else 3000
        if len(negix)>cap:negix=rng.choice(negix,cap,replace=False)
        ix=np.r_[posix,negix];selected[kind]=(rr[ix],yy[ix])
    mf=MotionFeatures(name,nodes,edges,TRAIN,R/'joint_cache');data={}
    for kind,(rr,yy) in selected.items():
        data[kind+'_x']=mf.rows(kind,rr) if len(rr) else np.empty((0,0),np.float32);data[kind+'_y']=yy
    tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,**data);os.replace(tmp,path)
    report={'movie':name,'group':group,'matched_nodes':len(matched),'edge_n':len(data['edge_y']),'edge_positive':int(data['edge_y'].sum()),'fork_n':len(data['fork_y']),'fork_positive':int(data['fork_y'].sum()),'seconds':time.time()-started,'feature_version':VERSION,'label_version':2}
    marker.write_text(json.dumps(report));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args();DEST.mkdir(exist_ok=True);split=json.loads((R/'events/split.json').read_text());(DEST/'split.json').write_text(json.dumps(split,indent=2));pending={n:g for g in ['train','calibration'] for n in split[g]}
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
        jobs={}
        while pending or jobs:
            for name,group in list(pending.items()):
                source=R/('training_baseline_graphs' if group=='train' else 'prior_graphs_B2')/(name+'.json')
                if source.exists() and len(jobs)<a.workers*2:
                    jobs[pool.submit(prepare,name,group)]=name;del pending[name]
            done=[f for f in jobs if f.done()]
            for f in done:
                row=f.result();print('MOTION_PREPARED',json.dumps(row),flush=True);del jobs[f]
            time.sleep(3)
    (DEST/'ready.json').write_text(json.dumps({'ready':True,'movies':sum(len(split[g]) for g in ['train','calibration'])}))
