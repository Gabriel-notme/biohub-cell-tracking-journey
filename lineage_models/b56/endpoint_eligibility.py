"""Geometry-only upper bound on possible endpoint additions; no labels are read."""
import numpy as np
from scipy.spatial import cKDTree
from refine_events import structure
from cell_event import chain

def endpoint_only_config(cfg):
    return (cfg.get('edge_consensus_gate',False) and cfg.get('edge_old_max',1)<0
            and cfg.get('edge_new_min',0)>1 and cfg.get('fork_threshold',0)>1
            and cfg.get('fork_veto',1)==0 and cfg.get('preserve_inherited_divisions',False)
            and cfg.get('endpoint_recovery',False)
            and not any(cfg.get(k,False) for k in ['fork_repair','endpoint_reassignment','division_reassignment','edge_low_confidence_pruning','division_local_assignment']))

def possible_endpoint_pairs(nodes,edges,cfg):
    out,prev,frames,pos=structure(nodes,edges);pairs=[]
    radius=min(cfg.get('endpoint_max_distance',8.),cfg.get('max_distance',14.))
    for t,sources in frames.items():
        targets=[n for n in frames.get(t+1,[]) if n not in prev and len(chain(n,out,pos))>=cfg.get('endpoint_min_future',3)]
        if not targets:continue
        tree=cKDTree([pos[n] for n in targets])
        for s in sources:
            if out.get(s):continue
            h=chain(s,prev,pos)
            if len(h)<cfg.get('endpoint_min_history',3):continue
            velocity=(h[0]-h[-1])/max(1,len(h)-1)
            for ix in tree.query_ball_point(pos[s],radius+1e-8):
                d=targets[ix]
                if np.linalg.norm(pos[d]-pos[s]-velocity)<=cfg.get('endpoint_motion_error',4.5)+1e-8:pairs.append((s,d))
    return pairs
