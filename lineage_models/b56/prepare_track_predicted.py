"""Mine detector-centered hard negatives, ignoring ambiguous sparse-label matches."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('POLARS_MAX_THREADS','2')
import numpy as np,json,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from prepare_events import graph
from refine_events import structure
from candidate_rows import candidate_rows
from joint_refine import JOINT_DEFAULT
from track_video import TrackMovie,sequences
from track_video_refine import geometries
R=Path('/workspace/biohub');D=R/os.environ.get('TRACK_PREDICTED_FOLDER','track_predicted');CHANNELS=int(os.environ.get('TRACK_VIDEO_CHANNELS','1'));B4=os.environ.get('TRACK_GRAPH_STAGE')=='B4';TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')

def prepare(name,group):
    start=time.time();path=D/(name+'.npz')
    if path.exists():return {'movie':name,'cached':True}
    folder=('reproduced_B4' if group=='calibration' else 'b4_training_graphs') if B4 else ('baseline_graphs' if group=='calibration' else 'training_baseline_graphs')
    if group=='train':folder=os.environ.get('TRACK_TRAIN_GRAPH_FOLDER',folder)
    raw=json.loads((R/folder/(name+'.json')).read_text());nodes={int(k):v for k,v in raw['nodes'].items()};edges=raw['edges']
    out,prev,frames,pos=structure(nodes,edges);gn,go,gp,gf,gpos=graph(TRAIN/(name+'.geff'));matched={};back={}
    for t,gs in gf.items():
        ns=frames.get(t,[])
        if not ns:continue
        dist=np.linalg.norm(np.asarray([gpos[g] for g in gs])[:,None]-np.asarray([pos[n] for n in ns])[None],axis=-1);ii,jj=linear_sum_assignment(dist)
        for i,j in zip(ii,jj):
            if dist[i,j]<=5.5:matched[ns[j]]=gs[i];back[gs[i]]=ns[j]
    er,fr=candidate_rows(nodes,edges,{**JOINT_DEFAULT,'max_candidates':int(os.environ.get('TRACK_MAX_CANDIDATES','5')),'max_distance':float(os.environ.get('TRACK_MAX_DISTANCE','14')),'fork_max_distance':16})
    rng=np.random.default_rng(int.from_bytes(name.encode()[-6:],'little')%2**32);selected={}
    for kind,candidates in [('edge',er),('fork',fr)]:
        rows=[]
        for row in candidates:
            s,*ds=map(int,row)
            if s not in matched or matched[s] not in go:continue
            g=matched[s];truth=go[g]
            if any(int(gn[d]['t'])!=int(gn[g]['t'])+1 for d in truth):continue
            resolved=[];ambiguous=False
            for d in ds:
                if d in matched:resolved.append(matched[d])
                elif min(np.linalg.norm(pos[d]-gpos[q]) for q in truth)<=7:ambiguous=True;break
                else:resolved.append(-1)
            if ambiguous:continue
            y=int(resolved[0] in truth) if kind=='edge' else int(len(truth)==2 and set(resolved)==set(truth))
            rows.append([s,*ds,y])
        rows=np.asarray(rows,np.int64).reshape(-1,3 if kind=='edge' else 4)
        positive=np.flatnonzero(rows[:,-1]);negative=np.flatnonzero(rows[:,-1]==0)
        cap=5500 if kind=='edge' else 4000
        if len(negative)>cap:
            parents=set(rows[positive,0]);hard=np.asarray([i for i in negative if rows[i,0] in parents],np.int64)
            rest=np.setdiff1d(negative,hard);remain=max(0,cap-len(hard));negative=np.r_[hard,rng.choice(rest,min(len(rest),remain),replace=False)]
        selected[kind]=rows[np.r_[positive,negative]]
    ids=sorted(nodes,key=lambda n:(nodes[n]['t'],n));lookup={n:i for i,n in enumerate(ids)};seq,motion=sequences(ids,nodes,out,prev)
    rows={k:np.c_[[[lookup[int(n)] for n in r[:-1]] for r in rr],rr[:,-1]].astype(np.int64) for k,rr in selected.items() if len(rr)}
    for k in ['edge','fork']:
        if k not in rows:rows[k]=np.empty((0,3 if k=='edge' else 4),np.int64)
    used=set()
    for k,rr in rows.items():
        for j in range(rr.shape[1]-1):used.update(seq[int(j>0),rr[:,j]].ravel().tolist())
    # Include complete chains for every referenced node, avoiding accidental sequence truncation.
    allix=sorted(used);compact={n:i for i,n in enumerate(allix)};oldseq=seq[:,allix]
    for a in oldseq.ravel():used.add(int(a))
    allix=sorted(used);compact={n:i for i,n in enumerate(allix)}
    # Only rows' anchor tracklets are consumed; unused bank nodes get self-padded tracks.
    newseq=np.broadcast_to(np.arange(len(allix))[None,:,None],(2,len(allix),seq.shape[2])).copy();newmotion=np.zeros((2,len(allix),seq.shape[2],4),np.float32);newmotion[:,:,0,3]=1
    anchors=set(int(n) for rr in rows.values() for n in rr[:,:-1].ravel())
    for n in anchors:
        for direction in range(2):
            newseq[direction,compact[n]]=[compact[int(q)] for q in seq[direction,n]];newmotion[direction,compact[n]]=motion[direction,n]
    posarr=np.asarray([pos[n] for n in ids]);metadata={}
    for k,rr in rows.items():
        metadata[k+'_geom']=geometries(rr[:,:-1],k,posarr,seq,motion)
        metadata[k]=np.c_[[[compact[int(n)] for n in r[:-1]] for r in rr],rr[:,-1]].astype(np.int64) if len(rr) else rr
    wanted=[ids[i] for i in allix];wi={n:i for i,n in enumerate(wanted)};movie=TrackMovie(TRAIN/(name+'.zarr'),CHANNELS);crops=np.empty((len(wanted),*((CHANNELS,) if CHANNELS>1 else ()),8,16,16),np.float16)
    for t,ns0 in sorted(frames.items()):
        ns=[n for n in ns0 if n in wi]
        for offset in range(0,len(ns),512):
            chosen=ns[offset:offset+512];crops[[wi[n] for n in chosen]]=movie.crops(t,[[nodes[n][k] for k in ['z','y','x']] for n in chosen])
    np.save(D/(name+'_crops.npy'),crops)
    phase=np.asarray([len(go.get(matched.get(n,-1),[]))==2 for n in wanted],np.float32)
    np.savez_compressed(path,**metadata,seq=newseq,motion=newmotion,phase=phase,ids=np.asarray(wanted))
    return {'movie':name,'group':group,'nodes':len(wanted),'edges':len(rows['edge']),'forks':len(rows['fork']),'edge_positive':int(rows['edge'][:,-1].sum()),'fork_positive':int(rows['fork'][:,-1].sum()),'seconds':time.time()-start}

if __name__=='__main__':
    D.mkdir(exist_ok=True)
    while not (R/'track_data/split.json').exists():time.sleep(5)
    split=json.loads((R/'track_data/split.json').read_text());(D/'split.json').write_text(json.dumps(split))
    pending={n:g for g in ['calibration','train'] for n in split[g]};reports=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn')) as pool:
        active={}
        while pending or active:
            for name,group in list(pending.items()):
                folder=('reproduced_B4' if group=='calibration' else 'b4_training_graphs') if B4 else ('baseline_graphs' if group=='calibration' else 'training_baseline_graphs')
                source=R/folder/(name+'.json')
                if source.exists() and len(active)<16:active[pool.submit(prepare,name,group)]=name;del pending[name]
            for f in [f for f in active if f.done()]:
                report=f.result();reports.append(report);print('PREDICTED_TRACK_PREPARED',json.dumps(report),flush=True);del active[f]
                (D/'preparation.json').write_text(json.dumps(reports))
            time.sleep(5)
    (D/'ready.json').write_text(json.dumps({'movies':len(reports)}));print('PREDICTED_TRACK_DATA_READY',flush=True)
