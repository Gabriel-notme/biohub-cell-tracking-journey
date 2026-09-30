from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('POLARS_MAX_THREADS','2')
import json,time,multiprocessing as mp,numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.spatial import cKDTree
from prepare_events import graph
from cell_event import SCALE
from trajectory_model import AppearanceMovie,STEPS,track_nodes,sequence_features
R=Path('/workspace/biohub');TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train');SOURCE=R/'joint_aligned';DEST=R/'trajectory_data'

def prepare(name):
    marker=DEST/(name+'_ready.json')
    if marker.exists():return json.loads(marker.read_text())
    started=time.time();nodes,out,prev,frames,pos=graph(TRAIN/(name+'.geff'))
    with np.load(SOURCE/(name+'.npz')) as d:data={k:d[k] for k in d.files}
    with np.load(R/'joint_data'/(name+'.npz')) as d:original_count=len(d['source_ids'])
    with np.load(R/'training_detector_candidates'/(name+'.npz')) as d:det={int(k):d[k] for k in d.files}
    trees={t:cKDTree(c*SCALE) for t,c in det.items() if len(c)};aligned=dict(pos)
    for t,ns in frames.items():
        if t not in trees:continue
        proposals=[]
        for n in ns:
            dist,j=trees[t].query(pos[n])
            if dist<=4.5:proposals.append((dist,n,int(j)))
        used=set()
        for dist,n,j in sorted(proposals):
            if j in used:continue
            used.add(j);aligned[n]=det[t][j]*SCALE
    gt_trees=[{t:cKDTree([pp[n] for n in ns]) for t,ns in frames.items()} for pp in [pos,aligned]]
    requests={};sequences=[];seq_index={}
    def request(t,p):
        coord=p/SCALE;key=(int(t),*np.rint(coord*16).astype(int).tolist())
        if key not in requests:requests[key]=(int(t),coord)
        return key
    def sequence(s,point,t,isparent,usealigned):
        key=(int(s) if isparent else -1,int(t),*np.rint(point*16).astype(int).tolist(),isparent,usealigned)
        if key in seq_index:return seq_index[key]
        pp=aligned if usealigned else pos
        if isparent:ids=track_nodes(s,prev,nodes);points=[pp[n] for n in ids];times=[nodes[n]['t'] for n in ids]
        else:
            dist,j=gt_trees[int(usealigned)][t].query(point)
            if dist<.25:
                n=frames[t][j];ids=track_nodes(n,out,nodes);points=[pp[n] for n in ids];times=[nodes[n]['t'] for n in ids]
            else:
                points=[point];times=[t]
                for dt in range(1,STEPS):
                    if t+dt not in trees:break
                    distance,k=trees[t+dt].query(points[-1])
                    if distance>8:break
                    points.append(det[t+dt][k]*SCALE);times.append(t+dt)
        keys=[request(tt,p) for tt,p in zip(times,points)];idx=len(sequences);seq_index[key]=idx;sequences.append((points,keys));return idx
    indices={}
    for kind in ['edge','fork']:
        rr=data[kind+'_rows'];rel=data[kind+'_rel'];rows=[]
        for row,relative in zip(rr,rel):
            ix=int(row[0]);s=int(data['source_ids'][ix]);t=int(nodes[s]['t']);al=ix>=original_count;pp=aligned if al else pos;origin=pp[s]
            seqs=[sequence(s,origin,t,True,al)]
            seqs.extend(sequence(s,origin+delta,t+1,False,al) for delta in relative[1:]);rows.append(seqs)
        indices[kind]=np.asarray(rows,np.int64)
    movie=AppearanceMovie(TRAIN/(name+'.zarr'));descriptors=movie.descriptors(requests)
    vectors=np.stack([sequence_features(points,descriptors,keys) for points,keys in sequences])
    metadata={}
    for kind in ['edge','fork']:
        np.save(DEST/(name+'_'+kind+'_x.npy'),vectors[indices[kind]].astype(np.float32));metadata[kind+'_y']=data[kind+'_rows'][:,-1];metadata[kind+'_geom']=data[kind+'_geom']
    np.savez_compressed(DEST/(name+'.npz'),**metadata)
    report={'movie':name,'sequences':len(sequences),'image_points':len(requests),'edges':len(indices['edge']),'forks':len(indices['fork']),'seconds':round(time.time()-started,1)};marker.write_text(json.dumps(report));return report

if __name__=='__main__':
    while not (SOURCE/'ready.json').exists():time.sleep(10)
    DEST.mkdir(exist_ok=True);split=json.loads((SOURCE/'split.json').read_text());(DEST/'split.json').write_text(json.dumps(split,indent=2));reports=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(prepare,n) for group in ['calibration','train'] for n in split[group]]
        for job in as_completed(jobs):
            row=job.result();reports.append(row);print('TRAJECTORY_PREPARED',json.dumps(row),flush=True);(DEST/'preparation.json').write_text(json.dumps(reports,indent=2))
    (DEST/'ready.json').write_text(json.dumps({'ready':True,'movies':len(reports)}));print('TRAJECTORY_DATA_READY',flush=True)
