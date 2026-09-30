from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('POLARS_MAX_THREADS','2')
import numpy as np,json,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from itertools import combinations
from scipy.spatial import cKDTree
from prepare_events import graph
from cell_event import SCALE,chain,edge_geometry,fork_geometry
from joint_model import JointMovie,SHAPE
R=Path('/workspace/biohub');TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train');SOURCE=R/'joint_mined';DEST=R/'joint_aligned'

def align(name):
    marker=DEST/(name+'_aligned.json')
    if marker.exists():return json.loads(marker.read_text())
    with np.load(SOURCE/(name+'.npz')) as d:data={k:d[k] for k in d.files}
    with np.load(R/'training_detector_candidates'/(name+'.npz')) as d:det={int(k):d[k] for k in d.files}
    nodes,out,prev,frames,pos=graph(TRAIN/(name+'.geff'));trees={t:cKDTree(c*SCALE) for t,c in det.items() if len(c)}
    matched={};assigned={};positions=dict(pos)
    for t,ns in frames.items():
        if t not in trees:continue
        proposals=[]
        for n in ns:
            distance,j=trees[t].query(pos[n])
            if distance<=4.5:proposals.append((distance,n,int(j)))
        used=set()
        for distance,n,j in sorted(proposals):
            if j in used:continue
            used.add(j);matched[n]=(t,j);assigned[(t,j)]=n;positions[n]=det[t][j]*SCALE
    count=len(data['source_ids']);sources=[int(s) for s in data['source_ids'] if int(s) in matched and all(d in matched for d in out.get(int(s),[]))]
    gt_trees={t:cKDTree([positions[n] for n in ns]) for t,ns in frames.items()}
    er=[];eg=[];erl=[];fr=[];fg=[];frl=[];child=[];phase=[]
    def future(t,j):
        known=assigned.get((t,j))
        if known is not None:return chain(known,out,positions)
        points=[det[t][j]*SCALE]
        for dt in range(1,4):
            if t+dt not in trees:break
            distance,k=trees[t+dt].query(points[-1])
            if distance>8:break
            points.append(det[t+dt][k]*SCALE)
        return np.asarray(points,np.float32)
    for i,s in enumerate(sources):
        index=count+i;t=int(nodes[s]['t']);valid=[d for d in out[s] if nodes[d]['t']==t+1];h=chain(s,prev,positions);origin=positions[s]
        target=np.full((2,3),np.nan,np.float32)
        for j,d in enumerate(valid):target[j]=positions[d]-origin
        child.append(target);phase.append(float(len(valid)==2))
        true=[chain(d,out,positions) for d in valid];false=[]
        near=gt_trees[t+1].query_ball_point(origin,18)
        rivals=[frames[t+1][j] for j in near if frames[t+1][j] not in valid and frames[t+1][j] in prev]
        rivals.sort(key=lambda n:np.linalg.norm(positions[n]-origin));false.extend(chain(n,out,positions) for n in rivals[:4])
        if t+1 in trees:
            near=trees[t+1].query_ball_point(origin,16)
            near=[j for j in near if assigned.get((t+1,j)) not in valid and min(np.linalg.norm(det[t+1][j]*SCALE-positions[d]) for d in valid)>4.5]
            near.sort(key=lambda j:np.linalg.norm(det[t+1][j]*SCALE-origin));false.extend(future(t+1,j) for j in near[:4])
        for ff,label in [(f,1) for f in true]+[(f,0) for f in false]:
            er.append([index,label]);eg.append(edge_geometry(h,ff));erl.append([[0,0,0],(ff[0]-origin).tolist()])
        pairs=[(aa,bb,0) for aa in true for bb in false if len(bb)>=2]
        if len(true)==2:pairs.append((*true,1))
        if len(false)>1:pairs.append((false[0],false[1],0))
        for aa,bb,label in pairs:
            fr.append([index,label]);fg.append(fork_geometry(h,aa,bb));frl.append([[0,0,0],(aa[0]-origin).tolist(),(bb[0]-origin).tolist()])
    oldbank=np.load(SOURCE/(name+'_patches.npy'),mmap_mode='r');context=oldbank.shape[1]
    bank=np.lib.format.open_memmap(DEST/(name+'_patches.npy'),mode='w+',dtype=np.float16,shape=(count+len(sources),context,*SHAPE));bank[:count]=oldbank
    movie=JointMovie(TRAIN/(name+'.zarr'),context)
    for i,s in enumerate(sources):bank[count+i]=movie.patch(nodes[s]['t'],positions[s]/SCALE)
    bank.flush();del bank,oldbank
    for kind,rows,geom,rel in [('edge',er,eg,erl),('fork',fr,fg,frl)]:
        gd=16 if kind=='edge' else 28;n=2 if kind=='edge' else 3
        data[kind+'_rows']=np.r_[data[kind+'_rows'],np.asarray(rows,np.float32).reshape(-1,2)]
        data[kind+'_geom']=np.r_[data[kind+'_geom'],np.asarray(geom,np.float32).reshape(-1,gd)]
        data[kind+'_rel']=np.r_[data[kind+'_rel'],np.asarray(rel,np.float32).reshape(-1,n,3)]
    data['children']=np.r_[data['children'],np.asarray(child,np.float32).reshape(-1,2,3)];data['phase']=np.r_[data['phase'],np.asarray(phase,np.float32)];data['source_ids']=np.r_[data['source_ids'],np.asarray(sources,np.int64)]
    np.savez_compressed(DEST/(name+'.npz'),**data);report={'movie':name,'added_sources':len(sources),'added_edges':len(er),'added_forks':len(fr),'added_divisions':int(sum(phase))};marker.write_text(json.dumps(report));return report

if __name__=='__main__':
    while not (SOURCE/'ready.json').exists():time.sleep(10)
    DEST.mkdir(exist_ok=True);split=json.loads((SOURCE/'split.json').read_text());(DEST/'split.json').write_text(json.dumps(split,indent=2))
    for name in split['calibration']:
        for suffix in ['.npz','_patches.npy']:
            p=DEST/(name+suffix)
            if not p.exists():p.symlink_to(SOURCE/(name+suffix))
    reports=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn')) as pool:
        for job in as_completed([pool.submit(align,n) for n in split['train']]):
            row=job.result();reports.append(row);print('JOINT_ALIGNED',json.dumps(row),flush=True);(DEST/'alignment_report.json').write_text(json.dumps(reports,indent=2))
    (DEST/'ready.json').write_text(json.dumps({'ready':True,'movies':len(reports)}));print('JOINT_ALIGNMENT_READY',flush=True)
