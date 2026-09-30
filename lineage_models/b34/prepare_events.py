from pathlib import Path
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse, json, time, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.spatial import cKDTree
from cell_event import Movie,SCALE,chain,edge_geometry,fork_geometry

def graph(path):
    import tracksdata as td
    g=td.graph.IndexedRXGraph.from_geff(path)
    if isinstance(g,tuple):g=g[0]
    nodes={int(n['node_id']):n for n in g.node_attrs().iter_rows(named=True)}
    out={};prev={};frames={}
    for e in g.edge_attrs().iter_rows(named=True):
        s,d=int(e['source_id']),int(e['target_id']);out.setdefault(s,[]).append(d);prev[d]=s
    for n,v in nodes.items():frames.setdefault(int(v['t']),[]).append(n)
    pos={n:np.array([v['z'],v['y'],v['x']],np.float32)*SCALE for n,v in nodes.items()}
    return nodes,out,prev,frames,pos

def prepare_one(args):
    root,dst,name,split,*extra=args;context=extra[0] if extra else 5;root=Path(root);dst=Path(dst);start=time.time()
    if (dst/(name+'.npz')).exists():return {'movie':name,'cached':True}
    rng=np.random.default_rng(int.from_bytes(name.encode()[-6:],'little')%2**32)
    nodes,out,prev,frames,pos=graph(root/(name+'.geff'))
    edge_rows=[];edge_geom=[];fork_rows=[];fork_geom=[]
    trees={t:cKDTree([pos[n] for n in ids]) for t,ids in frames.items()}
    # Stratified temporal sample, retaining every annotated mitosis.
    sources=list(out)
    if len(sources)>650:
        selected=set(rng.choice(sources,650,replace=False).tolist())
        selected.update(s for s,d in out.items() if len(d)==2)
        sources=sorted(selected)
    for s in sources:
        t=int(nodes[s]['t']);targets=frames.get(t+1,[])
        if not targets:continue
        h=chain(s,prev,pos)
        for d in out[s]:
            if int(nodes[d]['t'])!=t+1:continue
            edge_rows.append((s,d,1));edge_geom.append(edge_geometry(h,chain(d,out,pos)))
        near=trees[t+1].query_ball_point(pos[s],15)
        negatives=[targets[j] for j in near if targets[j] not in out[s] and targets[j] in prev]
        negatives.sort(key=lambda d:np.linalg.norm(pos[d]-pos[s]))
        for d in negatives[:4]:
            edge_rows.append((s,d,0));edge_geom.append(edge_geometry(h,chain(d,out,pos)))
    # Fork classifier sees both genuine events and close, real competing cells.
    positive=[];negative=[]
    for s,children in out.items():
        t=int(nodes[s]['t']);targets=frames.get(t+1,[])
        if not targets or not children:continue
        valid=[d for d in children if int(nodes[d]['t'])==t+1]
        if len(valid)==2:positive.append((s,valid[0],valid[1],1))
        if not valid:continue
        a=valid[0]
        near=trees[t+1].query_ball_point(pos[s],16)
        rivals=[targets[j] for j in near if targets[j] not in children and targets[j] in prev]
        rivals.sort(key=lambda d:np.linalg.norm(pos[d]-pos[s]))
        for b in rivals[:2]:negative.append((s,a,b,0))
    if len(negative)>200:
        negative=[negative[i] for i in rng.choice(len(negative),200,replace=False)]
    for s,a,b,label in positive+negative:
        fork_rows.append((s,a,b,label))
        fork_geom.append(fork_geometry(chain(s,prev,pos),chain(a,out,pos),chain(b,out,pos)))
    used=set(n for r in edge_rows for n in r[:2])|set(n for r in fork_rows for n in r[:3])
    ids=sorted(used,key=lambda n:(int(nodes[n]['t']),n));lookup={n:i for i,n in enumerate(ids)}
    bank=np.lib.format.open_memmap(dst/(name+'_patches.npy'),mode='w+',dtype=np.float16,shape=(len(ids),context,12,24,24))
    movie=Movie(root/(name+'.zarr'),context=context)
    phase=np.zeros(len(ids),np.float32)
    for i,n in enumerate(ids):
        v=nodes[n];bank[i]=movie.patch(v['t'],[v['z'],v['y'],v['x']])
        phase[i]=float(len(out.get(n,[]))==2)
    bank.flush();del bank
    er=np.asarray([(lookup[s],lookup[d],y) for s,d,y in edge_rows],np.int32).reshape(-1,3)
    fr=np.asarray([(lookup[s],lookup[a],lookup[b],y) for s,a,b,y in fork_rows],np.int32).reshape(-1,4)
    np.savez_compressed(dst/(name+'.npz'),edges=er,edge_geom=np.asarray(edge_geom,np.float32).reshape(-1,16),
        forks=fr,fork_geom=np.asarray(fork_geom,np.float32).reshape(-1,28),phase=phase,node_ids=np.asarray(ids),split=split)
    return {'movie':name,'split':split,'nodes':len(ids),'edges':len(er),'edge_positive':int(er[:,-1].sum()),
        'forks':len(fr),'fork_positive':len(positive),'seconds':round(time.time()-start,1)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/kaggle/input/competitions/biohub-cell-tracking-during-development/train');p.add_argument('--out',default='/workspace/biohub/events');p.add_argument('--workers',type=int,default=6);p.add_argument('--context',type=int,default=5);args=p.parse_args()
    root=Path(args.root);dst=Path(args.out);dst.mkdir(parents=True,exist_ok=True)
    if (dst/'split.json').exists():splits=json.loads((dst/'split.json').read_text())
    else:
        metadata=[]
        preview_names={p.stem for p in (root.parent/'test').glob('*.zarr')}
        for path in sorted(root.glob('*.geff')):
            if path.stem in preview_names:continue
            n,o,pr,f,po=graph(path);metadata.append({'name':path.stem,'divisions':sum(len(x)==2 for x in o.values()),'nodes':len(n)})
        rng=np.random.default_rng(20260921);splits={'train':[],'calibration':[],'audit':[],'metadata':metadata}
        for prefix in sorted({r['name'].split('_')[0] for r in metadata}):
            for isdiv in [True,False]:
                names=[r['name'] for r in metadata if r['name'].split('_')[0]==prefix and (r['divisions']>0)==isdiv]
                rng.shuffle(names);ncal=6 if isdiv else 3;naudit=6 if isdiv else 3
                splits['calibration']+=names[:ncal];splits['audit']+=names[ncal:ncal+naudit];splits['train']+=names[ncal+naudit:]
        assert not(set(splits['train'])&set(splits['calibration']+splits['audit']))
        (dst/'split.json').write_text(json.dumps(splits,indent=2))
    print('SPLIT', {k:len(v) for k,v in splits.items()},flush=True)
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(prepare_one,(str(root),str(dst),name,split,args.context)) for split in ['calibration','audit','train'] for name in splits[split]]
        for job in as_completed(jobs):
            r=job.result();records.append(r);print('PREPARED',json.dumps(r),flush=True)
            (dst/'preparation.json').write_text(json.dumps(records,indent=2))
    print('EVENTS_READY',flush=True)
