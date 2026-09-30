"""Build identical sparse candidate rows without unused geometry arrays."""
import numpy as np
from scipy.spatial import cKDTree
from itertools import combinations
from refine_events import structure
from cell_event import chain

def candidate_rows(nodes,edges,config,include_forks=True):
    out,prev,frames,pos=structure(nodes,edges);edge=[];fork=[]
    histories={s:chain(s,prev,pos) for s in nodes};future_lengths={n:len(chain(n,out,pos)) for n in nodes} if include_forks else {}
    for t,sources in sorted(frames.items()):
        targets=frames.get(t+1,[])
        if not targets:continue
        tree=cKDTree([pos[d] for d in targets]);nearby=tree.query_ball_point(np.stack([pos[s] for s in sources]),config['max_distance'],return_sorted=False)
        for s,indices in zip(sources,nearby):
            old=out.get(s,[]);h=histories[s];velocity=(h[0]-h[-1])/max(1,len(h)-1);near=[targets[i] for i in indices];near.sort(key=lambda d:np.linalg.norm(pos[d]-pos[s]-.5*velocity));cand=list(dict.fromkeys(old+near[:config['max_candidates']]))
            if not cand:continue
            edge.extend((s,d) for d in cand)
            if not include_forks:continue
            eligible=[d for d in cand if np.linalg.norm(pos[d]-pos[s])<=config['fork_max_distance'] and future_lengths[d]>=config['fork_min_branch']]
            pairs={tuple(sorted(ds)) for ds in combinations(eligible,2) if np.linalg.norm(pos[ds[0]]-pos[ds[1]])<=20}
            if len(old)==2:pairs.add(tuple(sorted(old)))
            fork.extend((s,a,b) for a,b in sorted(pairs))
    return np.asarray(edge,np.int64).reshape(-1,2),np.asarray(fork,np.int64).reshape(-1,3)
