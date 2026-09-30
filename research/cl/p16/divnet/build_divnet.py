"""Build DivNet training/eval rows per movie -> /dev/shm/divnet/<movie>.npz
GT rows (from the annotation, like artifact prepare_events): every GT division (1) + rival negatives (p, true child, nearby
non-daughter cell with a parent) top-2 per parent, sampled to 200/movie (0).
Pool rows (P15 final graphs, dv_cand labels by official matching): cand P=1, N/X=0 (N sampled 250/movie, X 150, weight = inverse
rate), D=-1 (already-recovered duplicate: eval only); existing forks Ftp=1 / Ffp=0 (one row per fork; veto analysis).
For pool rows also b1-style node patches (context 5) + fork_geometry on the P15 graph, so b1 variants score identical rows."""
import os, sys, json, glob
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','BLOSC_NTHREADS']: os.environ[k]='1'
ART='/workspace/art_b56/artifact_bundle'; sys.path.insert(0,ART); sys.path.insert(0,'/workspace/cl/p16/divnet')
from collections import defaultdict
from multiprocessing import get_context
import numpy as np
OUT='/dev/shm/divnet'


def job(name):
    of=OUT+'/'+name+'.npz'
    if os.path.exists(of): return name, 'cached'
    import numcodecs.blosc; numcodecs.blosc.use_threads=False
    from scipy.spatial import cKDTree
    import prepare_events
    from cell_event import SCALE, Movie, chain, fork_geometry
    import divnet_lib as L
    rng=np.random.default_rng(int.from_bytes(name.encode()[-6:],'little')%2**32)
    rows=[]; meta=[]
    # ---- GT rows
    nodes,out,prev,frames,pos=prepare_events.graph('/workspace/data/train/%s.geff'%name)
    trees={t:cKDTree([pos[n] for n in ids]) for t,ids in frames.items()}
    positive=[];negative=[]
    for s,children in out.items():
        t=int(nodes[s]['t']);targets=frames.get(t+1,[])
        if not targets or not children:continue
        valid=[d for d in children if int(nodes[d]['t'])==t+1]
        if len(valid)==2:positive.append((s,valid[0],valid[1],1))
        if not valid:continue
        a=valid[0]; near=trees[t+1].query_ball_point(pos[s],16)
        rivals=[targets[j] for j in near if targets[j] not in children and targets[j] in prev]
        rivals.sort(key=lambda d:np.linalg.norm(pos[d]-pos[s]))
        for b in rivals[:2]:negative.append((s,a,b,0))
    if len(negative)>200: negative=[negative[i] for i in rng.choice(len(negative),200,replace=False)]
    zyx=lambda v:[float(v['z']),float(v['y']),float(v['x'])]
    for s,a,b,y in positive+negative:
        rows.append((int(nodes[s]['t']),zyx(nodes[s]),zyx(nodes[a]),zyx(nodes[b])))
        meta.append(dict(src='gt',lab='G%d'%y,y=y,typ=-1,p=s,a=a,b=b,w=1.0))
    ngt=len(rows)
    # ---- pool rows
    fs=glob.glob('/workspace/cl/nm/dv_cand/*__%s.json'%name); assert len(fs)==1, fs
    st=os.path.basename(fs[0]).split('__')[0]
    R=json.load(open(fs[0]))
    d=json.load(open('/workspace/cl/ps_p15_%s/graphs/%s.json'%(st,name)))
    pn={int(k):v for k,v in d['nodes'].items()}
    pout,pprev=defaultdict(list),{}
    for e in d['edges']:
        u,v=int(e['source_id']),int(e['target_id']); pout[u].append(v); pprev[v]=u
    ppos={n:np.array([v['z'],v['y'],v['x']],np.float32)*SCALE for n,v in pn.items()}
    groups=defaultdict(list); seenf=set()
    for r in R:
        if r['src']=='cand':
            L0=r['lab'][0]
            if L0=='U': continue
            groups[L0].append(r)
        else:
            if r['lab'] not in ('Ftp','Ffp'): continue
            k=(r['p'],min(r['a'],r['b']),max(r['a'],r['b']))
            if k in seenf: continue
            seenf.add(k); groups[r['lab']].append(r)
    keep=[]
    for L0,cap in [('P',None),('D',None),('X',150),('N',250),('Ftp',None),('Ffp',None)]:
        g=groups.get(L0,[]); w=1.0
        if cap is not None and len(g)>cap:
            idx=rng.choice(len(g),cap,replace=False); w=len(g)/cap; g=[g[i] for i in sorted(idx)]
        for r in g: keep.append((L0,r,w))
    need=sorted({x for L0,r,w in keep for x in (r['p'],r['a'],r['b'])},key=lambda n:(pn[n]['t'],n)); nix={n:i for i,n in enumerate(need)}
    mv=Movie('/workspace/data/train/%s.zarr'%name,context=5)
    B1X=mv.patches([pn[n]['t'] for n in need],[[pn[n][k] for k in 'zyx'] for n in need]) if need else np.zeros((0,5,12,24,24),np.float16)
    B1T=np.full((ngt+len(keep),3),-1,np.int32); B1G=np.zeros((ngt+len(keep),28),np.float32)
    for i,(L0,r,w) in enumerate(keep):
        p,a,b=r['p'],r['a'],r['b']
        rows.append((int(pn[p]['t']),zyx(pn[p]),zyx(pn[a]),zyx(pn[b])))
        y={'P':1,'N':0,'X':0,'D':-1,'Ftp':1,'Ffp':0}[L0]
        meta.append(dict(src=r['src'],lab=r['lab'],y=y,typ=int(r['typ']),p=p,a=a,b=b,w=w,q=r.get('q')))
        B1T[ngt+i]=[nix[p],nix[a],nix[b]]
        B1G[ngt+i]=fork_geometry(chain(p,pprev,ppos),chain(a,pout,ppos),chain(b,pout,ppos))
    arr,low,high=L.load_movie('/workspace/data/train/%s.zarr'%name)
    fc=L.FrameCache(arr,low,high)
    X,rel=L.crops(fc,rows)
    np.savez(of+'.tmp.npz',X=X,rel=rel,y=np.array([m['y'] for m in meta],np.int8),w=np.array([m['w'] for m in meta],np.float32),
             typ=np.array([m['typ'] for m in meta],np.int8),src=np.array([m['src'] for m in meta]),lab=np.array([m['lab'] for m in meta]),
             ids=np.array([[m['p'],m['a'],m['b']] for m in meta],np.int64),q=np.array([-1 if m.get('q') is None else m['q'] for m in meta],np.int64),
             B1X=B1X,B1T=B1T,B1G=B1G,set=st)
    os.replace(of+'.tmp.npz',of)
    return name,len(meta)

if __name__=='__main__':
    os.makedirs(OUT,exist_ok=True)
    names=sorted(os.path.basename(f)[:-5] for f in glob.glob('/workspace/data/train/*.geff'))
    if len(sys.argv)>1: names=names[:int(sys.argv[1])]
    with get_context('spawn').Pool(56,maxtasksperchild=2) as p:
        for r in p.imap_unordered(job,names): print(r,flush=True)
    print('BUILD_DONE',flush=True)
