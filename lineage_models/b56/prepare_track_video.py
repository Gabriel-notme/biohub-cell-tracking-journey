"""Prepare only annotated transition labels; never label arbitrary cells as negatives."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('POLARS_MAX_THREADS','2')
import numpy as np,json,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.spatial import cKDTree
from itertools import combinations
from prepare_events import graph
from cell_event import SCALE,chain,edge_geometry,fork_geometry
from track_video import TrackMovie,sequences
R=Path('/workspace/biohub');D=R/os.environ.get('TRACK_DATA_FOLDER','track_data');CHANNELS=int(os.environ.get('TRACK_VIDEO_CHANNELS','1'));TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')

def prepare(name):
    start=time.time();path=D/(name+'.npz')
    if path.exists():return {'movie':name,'cached':True}
    nodes,out,prev,frames,pos=graph(TRAIN/(name+'.geff'));ids=sorted(nodes,key=lambda n:(nodes[n]['t'],n));ix={n:i for i,n in enumerate(ids)}
    trees={t:cKDTree([pos[n] for n in ns]) for t,ns in frames.items()};er=[];fr=[];eg=[];fg=[];phase=[]
    rng=np.random.default_rng(int.from_bytes(name.encode()[-6:],'little')%2**32)
    for s,ds in out.items():
        t=int(nodes[s]['t']);valid=[d for d in ds if nodes[d]['t']==t+1]
        if not valid or t+1 not in trees:continue
        near=[frames[t+1][j] for j in trees[t+1].query_ball_point(pos[s],20)]
        rivals=sorted([d for d in near if d not in ds and d in prev],key=lambda d:np.linalg.norm(pos[d]-pos[s]))[:5]
        h=chain(s,prev,pos)
        # All edges remain available, but redundant ordinary transitions are subsampled.
        if len(valid)==2 or rng.random()<.45:
            for d in valid+rivals[:3]:
                er.append([ix[s],ix[d],int(d in valid)]);eg.append(edge_geometry(h,chain(d,out,pos)))
        if len(valid)==2 or rng.random()<.10:
            pairs=set(tuple(sorted([a,b])) for a in valid for b in rivals[:3])
            if len(valid)==2:pairs.add(tuple(sorted(valid)))
            if len(rivals)>1:pairs.add(tuple(sorted(rivals[:2])))
            for a,b in sorted(pairs):
                fr.append([ix[s],ix[a],ix[b],int(set([a,b])==set(valid) and len(valid)==2)])
                fg.append(fork_geometry(h,chain(a,out,pos),chain(b,out,pos)))
    movie=TrackMovie(TRAIN/(name+'.zarr'),CHANNELS);crops=np.empty((len(ids),*((CHANNELS,) if CHANNELS>1 else ()),8,16,16),np.float16)
    for t,ns in sorted(frames.items()):
        crops[[ix[n] for n in ns]]=movie.crops(t,[[nodes[n][k] for k in ['z','y','x']] for n in ns])
    seq,motion=sequences(ids,nodes,out,prev)
    np.save(D/(name+'_crops.npy'),crops)
    np.savez_compressed(path,edge=np.asarray(er,np.int64).reshape(-1,3),fork=np.asarray(fr,np.int64).reshape(-1,4),
        edge_geom=np.asarray(eg,np.float32).reshape(-1,16),fork_geom=np.asarray(fg,np.float32).reshape(-1,28),
        seq=seq,motion=motion,phase=np.asarray([len(out.get(n,[]))==2 for n in ids],np.float32),ids=np.asarray(ids))
    return {'movie':name,'nodes':len(ids),'edges':len(er),'forks':len(fr),'divisions':sum(r[-1] for r in fr),'seconds':time.time()-start}

if __name__=='__main__':
    while not (R/'data_ready.json').exists():time.sleep(5)
    D.mkdir(exist_ok=True);split=json.loads((R/'events/split.json').read_text())
    import hashlib
    fresh=[]
    for embryo in ['44b6','6bba']:
        candidates=[n for n in split['train'] if n.startswith(embryo)]
        fresh.extend(sorted(candidates,key=lambda n:hashlib.sha256(('B56-audit-20260923:'+n).encode()).hexdigest())[:16])
    split['new_audit']=fresh;split['train']=[n for n in split['train'] if n not in fresh]
    split['validation_limitations']='New 32-movie audit excluded from all B5/B6 fitting and tuning; inherited B3/B4 and public detector saw these movies. Old audit is exposed historical evidence, not unseen validation.'
    (D/'split.json').write_text(json.dumps(split,indent=2))
    # Keep the old audit out of new training and model selection; its prior exposure is explicitly recorded.
    names=split['calibration']+split['train'];reports=[]
    with ProcessPoolExecutor(max_workers=12,mp_context=mp.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(prepare,n) for n in names]):
            r=f.result();reports.append(r);print('TRACK_PREPARED',json.dumps(r),flush=True)
            (D/'preparation.json').write_text(json.dumps(reports))
    (D/'ready.json').write_text(json.dumps({'movies':len(reports)}));print('TRACK_DATA_READY',flush=True)
