from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('POLARS_MAX_THREADS','2')
import numpy as np,json,time,argparse,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.spatial import cKDTree
from itertools import combinations
from prepare_events import graph
from cell_event import SCALE,chain,edge_geometry,fork_geometry
from joint_model import JointMovie,SHAPE,SPACING

def prepare_job(job):
    name,root,dst,context,group=job;root=Path(root);dst=Path(dst);start=time.time()
    if (dst/(name+'.npz')).exists():return {'movie':name,'cached':True}
    nodes,out,prev,frames,pos=graph(root/(name+'.geff'))
    rng=np.random.default_rng(int.from_bytes(name.encode()[-6:],'little')%2**32)
    trees={t:cKDTree([pos[n] for n in ns]) for t,ns in frames.items()}
    forks={s for s,ds in out.items() if len(ds)==2}
    # Every division plus neighboring time points, hard crowded parents, and temporal coverage.
    priority=set(forks)
    for source in forks:
        n=source
        for _ in range(4):
            if n not in prev:break
            n=prev[n];priority.add(n)
        frontier=out[source]
        for _ in range(4):
            priority.update(frontier);frontier=[d for n in frontier for d in out.get(n,[])]
    eligible=[s for s,ds in out.items() if any(int(nodes[d]['t'])==int(nodes[s]['t'])+1 for d in ds)]
    priority.update(rng.choice(eligible,min(320,len(eligible)),replace=False).tolist())
    sources=sorted(priority & set(eligible),key=lambda n:(nodes[n]['t'],n));lookup={n:i for i,n in enumerate(sources)}
    e_rows=[];e_geom=[];e_rel=[];f_rows=[];f_geom=[];f_rel=[];children=np.full((len(sources),2,3),np.nan,np.float32);phases=[]
    for s in sources:
        t=int(nodes[s]['t']);targets=frames.get(t+1,[]);valid=[d for d in out[s] if int(nodes[d]['t'])==t+1]
        phases.append(float(len(valid)==2))
        for j,d in enumerate(valid):children[lookup[s],j]=pos[d]-pos[s]
        h=chain(s,prev,pos)
        near=[targets[j] for j in trees[t+1].query_ball_point(pos[s],18)]
        rivals=sorted([d for d in near if d not in valid and d in prev],key=lambda d:np.linalg.norm(pos[d]-pos[s]))[:5]
        for d in valid+rivals:
            e_rows.append([lookup[s],float(d in valid)]);e_geom.append(edge_geometry(h,chain(d,out,pos)));e_rel.append([[0,0,0],(pos[d]-pos[s]).tolist()])
        pairs=set(tuple(sorted([a,b])) for a in valid for b in rivals[:3])
        if len(valid)==2:pairs.add(tuple(sorted(valid)))
        # Competing pair without the true continuation is a critical negative for reassignments.
        if len(rivals)>1:pairs.add(tuple(sorted(rivals[:2])))
        for a,b in sorted(pairs):
            f_rows.append([lookup[s],float(set([a,b])==set(valid))]);f_geom.append(fork_geometry(h,chain(a,out,pos),chain(b,out,pos)));f_rel.append([[0,0,0],(pos[a]-pos[s]).tolist(),(pos[b]-pos[s]).tolist()])
    bank=np.lib.format.open_memmap(dst/(name+'_patches.npy'),mode='w+',dtype=np.float16,shape=(len(sources),context,*SHAPE))
    movie=JointMovie(root/(name+'.zarr'),context)
    for i,s in enumerate(sources):bank[i]=movie.patch(nodes[s]['t'],[nodes[s][k] for k in ['z','y','x']])
    bank.flush();del bank
    np.savez_compressed(dst/(name+'.npz'),edge_rows=np.asarray(e_rows,np.float32).reshape(-1,2),edge_geom=np.asarray(e_geom,np.float32).reshape(-1,16),edge_rel=np.asarray(e_rel,np.float32).reshape(-1,2,3),fork_rows=np.asarray(f_rows,np.float32).reshape(-1,2),fork_geom=np.asarray(f_geom,np.float32).reshape(-1,28),fork_rel=np.asarray(f_rel,np.float32).reshape(-1,3,3),children=children,phase=np.asarray(phases,np.float32),source_ids=np.asarray(sources))
    return {'movie':name,'group':group,'sources':len(sources),'edges':len(e_rows),'forks':len(f_rows),'divisions':int(sum(phases)),'seconds':round(time.time()-start,1)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',type=int,default=7);p.add_argument('--workers',type=int,default=8);p.add_argument('--out',default='/workspace/biohub/joint_data');a=p.parse_args()
    R=Path('/workspace/biohub');root=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train');dst=Path(a.out);dst.mkdir(exist_ok=True)
    while not (R/'data_ready.json').exists():time.sleep(10)
    split=json.loads((R/'events/split.json').read_text());(dst/'split.json').write_text(json.dumps(split,indent=2))
    records=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(prepare_job,(name,str(root),str(dst),a.context,g)) for g in ['calibration','train'] for name in split[g]]
        for job in as_completed(jobs):
            row=job.result();records.append(row);print('JOINT_PREPARED',json.dumps(row),flush=True);(dst/'preparation.json').write_text(json.dumps(records,indent=2))
    (dst/'ready.json').write_text(json.dumps({'ready':True,'context':a.context,'movies':len(records)}));print('JOINT_DATA_READY',flush=True)
