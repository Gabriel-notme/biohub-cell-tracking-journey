import sys, json, os
sys.path.insert(0,'/workspace/code')
os.environ.setdefault('POLARS_MAX_THREADS','2')
from evalx import load_gt, K
import numpy as np, zarr
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
TR='/workspace/data/train'
def one(name):
    gt, n_total = load_gt(name)
    na = gt.node_attrs(attr_keys=[K.NODE_ID,'t','z','y','x'])
    ids = na[K.NODE_ID].to_list(); t=np.array(na['t'].to_list()); z=np.array(na['z'].to_list()); y=np.array(na['y'].to_list()); x=np.array(na['x'].to_list())
    ea = gt.edge_attrs(); src=ea[K.EDGE_SOURCE].to_list(); dst=ea[K.EDGE_TARGET].to_list()
    shp = zarr.open(TR+'/'+name+'.zarr/0', mode='r').shape
    idx={n:i for i,n in enumerate(ids)}
    outd=defaultdict(int); ind=defaultdict(int)
    for s,d in zip(src,dst): outd[s]+=1; ind[d]+=1
    # components via union find
    par=list(range(len(ids)))
    def f(a):
        while par[a]!=a: par[a]=par[par[a]]; a=par[a]
        return a
    for s,d in zip(src,dst): par[f(idx[s])]=f(idx[d])
    comp=defaultdict(list)
    for i in range(len(ids)): comp[f(i)].append(i)
    clen=[]; cstart=[]; cend=[]
    for c,m in comp.items():
        ts=t[m]; clen.append(int(ts.max()-ts.min()+1)); cstart.append(int(ts.min())); cend.append(int(ts.max()))
    starts=[i for i,n in enumerate(ids) if ind[n]==0]
    ndiv=sum(1 for n in ids if outd[n]>=2)
    return dict(movie=name, T=shp[0], Z=shp[1], Y=shp[2], X=shp[3], n_gt=len(ids), n_edges=len(src), n_total=n_total,
        frac=len(ids)/n_total if n_total else None, t_min=int(t.min()), t_max=int(t.max()), n_frames=int(len(set(t.tolist()))),
        n_comp=len(comp), comp_len_med=float(np.median(clen)), comp_len_mean=float(np.mean(clen)), comp_full=float(np.mean([(a==0 and b==shp[0]-1) for a,b in zip(cstart,cend)])),
        start_t0=float(np.mean([a==0 for a in cstart])), end_last=float(np.mean([b==shp[0]-1 for b in cend])),
        n_roots=len(starts), root_t0=float(np.mean([t[i]==0 for i in starts])) if starts else None, ndiv=ndiv,
        z_lo=float(np.percentile(z,5)), z_hi=float(np.percentile(z,95)), y_lo=float(np.percentile(y,5)), y_hi=float(np.percentile(y,95)), x_lo=float(np.percentile(x,5)), x_hi=float(np.percentile(x,95)),
        per_frame_med=float(np.median(np.bincount(t.astype(int)))) )
if __name__=='__main__':
    names=sorted(p[:-5] for p in os.listdir(TR) if p.endswith('.geff'))
    with ProcessPoolExecutor(32) as ex: rows=list(ex.map(one,names))
    json.dump(rows, open('/workspace/runs/gt_stats.json','w'))
    print(len(rows))
