"""p19_frag: re-apply B5's own short-track output filter at the very end of the P-stage (GT-free, deletion only).
B5 (cloud_baseline.filter_short_track_components): a weakly connected component with < OUTPUT_MIN_TRACK_LEN=6 nodes and no fork
(no node with >= 2 children) is removed; adaptive rescue keeps one with >= 4 nodes, mean edge_prob >= 0.88 and mean edge
distance_um <= 3.0 (B5 only triggers the rescue when >= 10% of nodes would go and caps it at 1.2% of nodes; here rescue=1 applies
the keep-criteria without trigger/budget).  B5 ran the filter BEFORE its lineage stage and before the whole P-stage, both of which
cut edges and leave new short fragments that were never re-checked.
  scope='all'    : every short fork-free component of the final graph
  scope='pstage' : only components the P-stage created or modified, i.e. whose node set is not exactly a weakly connected component
                   of the B5 lineage graph (the P-stage input) - cut from a larger B5 component, merged, or containing new nodes
  scope='dup'    : only components of 1-2 nodes whose every node lies within `rad` um of a node of a long track (component >= 6
                   nodes or with a fork) in the same frame (duplicate detections of an already tracked cell)
  scope='iso'    : only isolated single nodes (B5 OUTPUT_PRUNE_ISOLATED / P-stage post_prune=2 re-applied at the very end)
Frame guard as in prune.py: a frame is never left empty."""
import json
from collections import defaultdict
import numpy as np
WANTS_META = True
B5 = {'hold36': '/workspace/runs/b5f_hold36', 'prev4': '/workspace/runs/b5f_prev4', 'audit32': '/workspace/sync3/runs/b5f_audit32',
      't127a': '/workspace/sync4/runs/b5f_t127a', 't127b': '/workspace/sync3/runs/b5f_t127b'}
S = np.array([1.625, 0.40625, 0.40625])
MINLEN, RESC_MIN, RESC_P, RESC_D = 6, 4, 0.88, 3.0


def _wcc(ids, edges):
    adj = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); adj[a].append(b); adj[b].append(a)
    seen = set(); C = []
    for n in ids:
        if n in seen: continue
        c = []; st = [n]; seen.add(n)
        while st:
            u = st.pop(); c.append(u)
            for v in adj.get(u, []):
                if v not in seen: seen.add(v); st.append(v)
        C.append(c)
    return C


def _xyz(v):
    return np.array([float(v['z']), float(v['y']), float(v['x'])]) * S


def frag_filter(nodes, edges, scope='all', rescue=0, rad=7.0, linp=None):
    out = defaultdict(int)
    for e in edges: out[int(e['source_id'])] += 1
    C = _wcc(list(nodes), edges)
    short = [c for c in C if len(c) < MINLEN and not any(out.get(u, 0) >= 2 for u in c)]
    st = defaultdict(int)
    if scope == 'pstage':
        lin = json.load(open(linp)); linF = {frozenset(c) for c in _wcc([int(k) for k in lin['nodes']], lin['edges'])}
        short = [c for c in short if frozenset(c) not in linF]
    elif scope == 'iso':  # B5 OUTPUT_PRUNE_ISOLATED / P-stage post_prune re-applied: edge-less single nodes left by later steps
        short = [c for c in short if len(c) == 1]
    elif scope == 'dup':
        from scipy.spatial import cKDTree
        sset = {u for c in short for u in c}; byt = defaultdict(list)
        for n in nodes:
            if n not in sset: byt[int(nodes[n]['t'])].append(n)
        trees = {t: cKDTree(np.stack([_xyz(nodes[u]) for u in us])) for t, us in byt.items()}
        keep = []
        for c in short:
            if len(c) > 2: continue
            if all(int(nodes[u]['t']) in trees and trees[int(nodes[u]['t'])].query_ball_point(_xyz(nodes[u]), rad) for u in c): keep.append(c)
        short = keep
    drop = set()
    if short:
        cid = {}
        for i, c in enumerate(short):
            for u in c: cid[u] = i
        ce = defaultdict(list)
        for e in edges:
            a = int(e['source_id'])
            if a in cid: ce[cid[a]].append(e)
        for i, c in enumerate(short):
            if rescue and len(c) >= RESC_MIN and ce[i]:
                pr = []; ds = []
                for e in ce[i]:
                    try: p = float(e.get('edge_prob', 0.0))
                    except (TypeError, ValueError): p = 0.0
                    if np.isfinite(p): pr.append(p)
                    try: dd = float(e.get('distance_um', np.nan))
                    except (TypeError, ValueError): dd = np.nan
                    if not np.isfinite(dd): dd = float(np.linalg.norm(_xyz(nodes[int(e['source_id'])]) - _xyz(nodes[int(e['target_id'])])))
                    ds.append(dd)
                if pr and np.mean(pr) >= RESC_P and np.mean(ds) <= RESC_D: st['rescued'] += 1; continue
            drop.update(c); st['del_comp'] += 1; st['del_size%d' % len(c)] += 1
    if drop:  # frame guard: never empty a frame
        ts = defaultdict(int)
        for n, v in nodes.items():
            if n not in drop: ts[int(v['t'])] += 1
        for n in sorted(drop):
            t = int(nodes[n]['t'])
            if ts[t] == 0: drop.discard(n); ts[t] += 1; st['frame_guard'] += 1
    st['removed_nodes'] = len(drop)
    if not drop: return nodes, edges, dict(st)
    return ({k: v for k, v in nodes.items() if k not in drop},
            [e for e in edges if int(e['source_id']) not in drop and int(e['target_id']) not in drop], dict(st))


def apply(nodes, edges, scope='all', rescue=0, rad=7.0, name=None, **kw):
    linp = B5[kw['set']] + '/working/lineage_graphs/%s.json' % name if scope == 'pstage' else None
    return frag_filter(nodes, edges, scope=scope, rescue=rescue, rad=rad, linp=linp)
