"""Build the new DivNet v2 training rows per movie -> /dev/shm/divnet2/<movie>.npz (see NOTES.md pre-registration).
PS = P15-final-graph forks labelled 'Fnc' in nm/dv_cand (pseudo-positive, target 0.8); PR = 2 nearest rival nodes per PS daughter
(target 0); HN = top-25 per type by b1dep among labelled N/X candidates of the real first-pass dc pool (dnlog of v1 run dnA)."""
import os, sys, json, glob
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','BLOSC_NTHREADS']: os.environ[k]='1'
sys.path.insert(0,'/workspace/cl/p16/divnet')
from collections import defaultdict
from multiprocessing import get_context
import numpy as np
OUT='/dev/shm/divnet2'; SCALE=np.array([1.625,0.40625,0.40625],np.float32); KHN=25


def job(name):
    of=OUT+'/'+name+'.npz'
    if os.path.exists(of): return name,'cached'
    import numcodecs.blosc; numcodecs.blosc.use_threads=False
    from scipy.spatial import cKDTree
    import divnet_lib as L
    fs=glob.glob('/workspace/cl/nm/dv_cand/*__%s.json'%name); assert len(fs)==1, fs
    st=os.path.basename(fs[0]).split('__')[0]
    R=json.load(open(fs[0]))
    d=json.load(open('/workspace/cl/ps_p15_%s/graphs/%s.json'%(st,name)))
    pn={int(k):v for k,v in d['nodes'].items()}
    pout,pprev=defaultdict(list),{}
    for e in d['edges']:
        u,v=int(e['source_id']),int(e['target_id']); pout[u].append(v); pprev[v]=u
    ppos={n:np.array([v['z'],v['y'],v['x']],np.float32)*SCALE for n,v in pn.items()}
    frames=defaultdict(list)
    for n,v in pn.items(): frames[int(v['t'])].append(n)
    trees={t:cKDTree(np.stack([ppos[n] for n in ids])) for t,ids in frames.items()}
    trip=[]; kind=[]; typ=[]
    # PS + PR
    seen=set(); seenr=set(); lab={}
    for r in R:
        if r['src']=='cand': lab[(r['p'],r['a'],r['b'])]=r['lab'][0]; continue
        if r['lab']!='Fnc': continue
        p,a,b=r['p'],r['a'],r['b']; k=(p,min(a,b),max(a,b))
        if k in seen: continue
        seen.add(k); trip.append((p,a,b)); kind.append('ps'); typ.append(-1)
        t=int(pn[p]['t']); ids=frames.get(t+1,[])
        if not ids: continue
        near=trees[t+1].query_ball_point(ppos[p],16.0)
        riv=[ids[j] for j in near if ids[j] not in pout[p] and ids[j] in pprev]
        riv.sort(key=lambda n:float(np.linalg.norm(ppos[n]-ppos[p])))
        for dd in (a,b):
            for rr in riv[:2]:
                kk=(p,dd,rr)
                if kk in seenr: continue
                seenr.add(kk); trip.append(kk); kind.append('pr'); typ.append(-1)
    nps=kind.count('ps'); npr=kind.count('pr')
    # HN from the real first-pass dc pool (v1 dnA dnlog; candidate set + fork_b1 are DivNet-independent)
    dl='/workspace/cl/p16/divnet/ps_dnA_%s/dnlog/%s_1.json'%(st,name); nhn=0
    if os.path.exists(dl):
        C=json.load(open(dl))
        for ty,tc in (('start',0),('stolen',1)):
            cc=[c for c in C if c['typ']==ty and lab.get((c['p'],c['a'],c['b'])) in ('N','X') and all(x in pn for x in (c['p'],c['a'],c['b']))]
            cc.sort(key=lambda c:-c['fork_b1'])
            for c in cc[:KHN]: trip.append((c['p'],c['a'],c['b'])); kind.append('hn'); typ.append(tc); nhn+=1
    zyx=lambda n:[float(pn[n]['z']),float(pn[n]['y']),float(pn[n]['x'])]
    rows=[(int(pn[p]['t']),zyx(p),zyx(a),zyx(b)) for p,a,b in trip]
    arr,low,high=L.load_movie('/workspace/data/train/%s.zarr'%name)
    fc=L.FrameCache(arr,low,high)
    if rows: X,rel=L.crops(fc,rows)
    else: X=np.zeros((0,5)+L.CROP,np.float16); rel=np.zeros((0,3,3),np.float32)
    np.savez(of+'.tmp.npz',X=X,rel=rel,kind=np.array(kind),typ=np.array(typ,np.int8),ids=np.array(trip,np.int64).reshape(-1,3),set=st)
    os.replace(of+'.tmp.npz',of)
    return name,nps,npr,nhn

if __name__=='__main__':
    os.makedirs(OUT,exist_ok=True)
    names=sorted(os.path.basename(f)[:-5] for f in glob.glob('/workspace/data/train/*.geff'))
    if len(sys.argv)>1: names=names[:int(sys.argv[1])]
    tot=np.zeros(3,int)
    with get_context('spawn').Pool(int(os.environ.get('NP','10')),maxtasksperchild=4) as p:
        for r in p.imap_unordered(job,names):
            print(r,flush=True)
            if len(r)==4: tot+=np.array(r[1:])
    print('TOTAL ps/pr/hn',tot.tolist()); print('BUILD_DONE',flush=True)
