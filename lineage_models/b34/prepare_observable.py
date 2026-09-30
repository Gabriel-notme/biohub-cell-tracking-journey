"""Train on observable detector-centered divisions, including one-frame delayed resolution.

Only training movies are augmented. A delayed positive requires both annotated daughters
to map to the SAME actual image detection before they become distinct detections.
"""
from pathlib import Path
import os,json,time,multiprocessing as mp
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('POLARS_MAX_THREADS','2')
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.spatial import cKDTree
from prepare_events import graph
from cell_event import SCALE,chain,edge_geometry,fork_geometry
from joint_model import JointMovie,SHAPE
R=Path('/workspace/biohub');TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train');SRC=R/'joint_aligned';DST=R/'joint_observable'

def prepare(name):
    marker=DST/(name+'_ready.json')
    if marker.exists():return json.loads(marker.read_text())
    nodes,out,prev,frames,pos=graph(TRAIN/(name+'.geff'))
    with np.load(SRC/(name+'.npz')) as z:data={k:z[k] for k in z.files}
    with np.load(R/'training_detector_candidates'/(name+'.npz')) as z:det={int(k):z[k]*SCALE for k in z.files}
    trees={t:cKDTree(v) for t,v in det.items() if len(v)}
    def match(n):
        t=int(nodes[n]['t'])
        if t not in trees:return None
        dd,j=trees[t].query(pos[n]);return (int(j),det[t][j]) if dd<=6 else None
    def future(n):
        ids=[n];cur=n
        for _ in range(3):
            ds=out.get(cur,[])
            if len(ds)!=1:break
            cur=ds[0];ids.append(cur)
        return np.asarray([match(k)[1] if match(k) is not None else pos[k] for k in ids],np.float32)
    new=[];ambiguous=set();delays=0
    for s,ds in out.items():
        if len(ds)!=2:continue
        t=int(nodes[s]['t']);m=match(s)
        if m is None:continue
        candidates=[(t,m[1],ds,chain(s,prev,pos),0)]
        ma,mb=match(ds[0]),match(ds[1])
        if ma is not None and mb is not None and ma[0]==mb[0] and all(len(out.get(d,[]))==1 for d in ds):
            nexts=[out[d][0] for d in ds]
            candidates.append((t+1,ma[1],nexts,np.vstack([ma[1],chain(s,prev,pos)[:3]]),1));ambiguous.update(ds)
        for tt,origin,children,history,delay in candidates:
            aa,bb=match(children[0]),match(children[1])
            if aa is None or bb is None or aa[0]==bb[0] or np.linalg.norm(aa[1]-bb[1])<2.5:continue
            if max(np.linalg.norm(aa[1]-origin),np.linalg.norm(bb[1]-origin))>16:continue
            true=[future(d) for d in children];true[0][0]=aa[1];true[1][0]=bb[1]
            near=trees[tt+1].query_ball_point(origin,16);near=[j for j in near if j not in [aa[0],bb[0]] and min(np.linalg.norm(det[tt+1][j]-p[0]) for p in true)>4.]
            near.sort(key=lambda j:np.linalg.norm(det[tt+1][j]-origin));false=[]
            for j in near[:6]:
                ff=[det[tt+1][j]]
                for dt in range(2,5):
                    if tt+dt not in trees:break
                    dd,k=trees[tt+dt].query(ff[-1])
                    if dd>8:break
                    ff.append(det[tt+dt][k])
                false.append(np.asarray(ff,np.float32))
            new.append((s,tt,origin,history,true,false,delay));delays+=delay
    count=len(data['source_ids']);er=[];eg=[];rel=[];fr=[];fg=[];fre=[];ch=[]
    for i,(s,t,origin,h,true,false,delay) in enumerate(new):
        ix=count+i;h=h.copy();h[0]=origin;ch.append([ff[0]-origin for ff in true])
        for ff,label in [(f,1) for f in true]+[(f,0) for f in false]:
            er.append([ix,label]);eg.append(edge_geometry(h,ff));rel.append([[0,0,0],ff[0]-origin])
        pairs=[(*true,1)]+[(a,b,0) for a in true for b in false]
        for a,b,y in pairs:fr.append([ix,y]);fg.append(fork_geometry(h,a,b));fre.append([[0,0,0],a[0]-origin,b[0]-origin])
    # At a merged observation, a single annotated daughter's continuation is not
    # a reliable negative for the physical division event of the merged blob.
    rr=data['fork_rows'];bad=np.array([int(data['source_ids'][int(ix)]) in ambiguous and y==0 for ix,y in rr])
    for key in ['fork_rows','fork_geom','fork_rel']:data[key]=data[key][~bad]
    if new:
        old=np.load(SRC/(name+'_patches.npy'),mmap_mode='r');bank=np.lib.format.open_memmap(DST/(name+'_patches.npy'),mode='w+',dtype=np.float16,shape=(count+len(new),7,*SHAPE));bank[:count]=old
        movie=JointMovie(TRAIN/(name+'.zarr'),7)
        for i,(_,t,origin,*_) in enumerate(new):bank[count+i]=movie.patch(t,origin/SCALE)
        bank.flush();del bank,old
    else:(DST/(name+'_patches.npy')).symlink_to(SRC/(name+'_patches.npy'))
    for kind,rr,gg,pp in [('edge',er,eg,rel),('fork',fr,fg,fre)]:
        dim=16 if kind=='edge' else 28;n=2 if kind=='edge' else 3
        data[kind+'_rows']=np.r_[data[kind+'_rows'],np.asarray(rr,np.float32).reshape(-1,2)]
        data[kind+'_geom']=np.r_[data[kind+'_geom'],np.asarray(gg,np.float32).reshape(-1,dim)]
        data[kind+'_rel']=np.r_[data[kind+'_rel'],np.asarray(pp,np.float32).reshape(-1,n,3)]
    data['source_ids']=np.r_[data['source_ids'],np.asarray([x[0] for x in new],np.int64)];data['children']=np.r_[data['children'],np.asarray(ch,np.float32).reshape(-1,2,3)];data['phase']=np.r_[data['phase'],np.ones(len(new),np.float32)]
    np.savez_compressed(DST/(name+'.npz'),**data)
    report={'movie':name,'additional_observable_divisions':len(new),'delayed_merged_divisions':delays,'removed_ambiguous_negatives':int(bad.sum())};marker.write_text(json.dumps(report));return report

if __name__=='__main__':
    DST.mkdir(exist_ok=True);split=json.loads((SRC/'split.json').read_text());(DST/'split.json').write_text(json.dumps(split,indent=2))
    for n in split['calibration']:
        for suffix in ['.npz','_patches.npy']:
            p=DST/(n+suffix)
            if not p.exists():p.symlink_to(SRC/(n+suffix))
    rows=[]
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(prepare,n) for n in split['train']]):
            row=f.result();rows.append(row);print('OBSERVABLE_PREPARED',json.dumps(row),flush=True);(DST/'preparation.json').write_text(json.dumps(rows,indent=2))
    (DST/'ready.json').write_text(json.dumps({'ready':True,'movies':len(rows),'additional_divisions':sum(r['additional_observable_divisions'] for r in rows),'delayed_divisions':sum(r['delayed_merged_divisions'] for r in rows)}))
