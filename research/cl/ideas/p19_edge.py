"""p19_edge: boundary / temporal stub deletion on the final P-stage graph (GT-free, deletion only, B5 constants).
  tstub : weakly connected components of <= 3 nodes, no fork, that touch the first or the last frame of the movie and have a node
          within 4.0 um (same frame) of a node of a longer track (component >= 6 nodes = B5 OUTPUT_MIN_TRACK_LEN): a duplicate stub of an
          existing cell cut by the temporal boundary -> remove the component.
  border: components < 6 nodes, no fork, made of volume-border nodes (z rounds to 0 or Z-1, or y/x within 2 voxels of the y/x edge):
          partial cells cut by the field of view. mode 'all' = every node on the border, 'any' = at least one node on the border.
  padd  : components < 6 nodes, no fork, that contain a P-stage addition (edge flag tb_ext / edge_link / long_link, node flag tb_ext /
          edge_link_node): additions whose host track was later cut, left as isolated fragments B5's short-track filter never saw.
Removal is always of whole components; the division / fork structure of other components is never touched."""
import json
import os
from collections import defaultdict
import numpy as np

WANTS_META = True
S = np.array([1.625, .40625, .40625])
MINLEN = 6              # B5 OUTPUT_MIN_TRACK_LEN
PF_E = ('tb_ext', 'edge_link', 'long_link')
PF_N = ('tb_ext', 'edge_link_node')


def _shape(zarr):
    try:
        for p in (os.path.join(zarr, '0', 'zarr.json'), os.path.join(zarr, '0', '.zarray')):
            if os.path.exists(p):
                return [int(v) for v in json.load(open(p))['shape']]
    except Exception:
        pass
    return [100, 64, 256, 256]


def _comps(nodes, edges):
    adj = defaultdict(list); out = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); adj[a].append(b); adj[b].append(a); out[a].append(b)
    cid = {}; comps = []
    for n in nodes:
        if n in cid: continue
        st = [n]; cid[n] = len(comps); cc = [n]
        while st:
            u = st.pop()
            for v in adj.get(u, []):
                if v not in cid: cid[v] = len(comps); st.append(v); cc.append(v)
        comps.append(cc)
    return comps, cid, out


def select(nodes, edges, tstub=0, tstub_max=3, tstub_rad=4.0, border=0, border_mode='all', border_yx=2.0, padd=0, padd_mode='any',
           shape=None):
    """returns (set of node ids to remove, stats). nodes: {int id: {t,z,y,x,...}} (voxel coordinates)"""
    from scipy.spatial import cKDTree
    shape = shape or [100, 64, 256, 256]
    T, Z, Y, X = shape
    comps, cid, out = _comps(nodes, edges)
    csize = [len(c) for c in comps]
    fork = [any(len(out.get(u, [])) >= 2 for u in c) for c in comps]
    drop = set(); st = defaultdict(int)
    if tstub:
        by_t = defaultdict(list)
        for n, v in nodes.items():
            if csize[cid[n]] >= MINLEN: by_t[int(v['t'])].append(n)
        trees = {}
        for ci, c in enumerate(comps):
            if csize[ci] > tstub_max or fork[ci]: continue
            ts = [int(nodes[u]['t']) for u in c]
            if min(ts) != 0 and max(ts) != T - 1: continue
            hit = False
            for u in c:
                t = int(nodes[u]['t'])
                if t not in trees:
                    ns = by_t.get(t, [])
                    trees[t] = cKDTree(np.array([[nodes[m]['z'], nodes[m]['y'], nodes[m]['x']] for m in ns], float) * S) if ns else None
                tr = trees[t]
                if tr is None: continue
                p = np.array([nodes[u]['z'], nodes[u]['y'], nodes[u]['x']], float) * S
                if tr.query_ball_point(p, tstub_rad - 1e-9): hit = True; break
            if hit:
                drop.update(c); st['ts_comp'] += 1; st['ts_nodes'] += len(c)
    if border:
        def isz(v):
            z = float(v['z'])
            return z < 0.5 or z >= Z - 1.5

        def isyx(v):
            y, x = float(v['y']), float(v['x'])
            return y <= border_yx or y >= Y - 1 - border_yx or x <= border_yx or x >= X - 1 - border_yx
        isb = {'all': lambda v: isz(v) or isyx(v), 'any': lambda v: isz(v) or isyx(v), 'allz': isz, 'allyx': isyx}[border_mode]
        for ci, c in enumerate(comps):
            if csize[ci] >= MINLEN or fork[ci]: continue
            b = [isb(nodes[u]) for u in c]
            if (border_mode != 'any' and all(b)) or (border_mode == 'any' and any(b)):
                new = set(c) - drop
                if new: st['bd_comp'] += 1; st['bd_nodes'] += len(new)
                drop.update(c)
    if padd:
        flagged = set(); synth = set(); ce = defaultdict(list)
        for e in edges:
            a = int(e['source_id']); pf = any(f in e for f in PF_E); ce[cid[a]].append(pf)
            if pf: flagged.add(a); flagged.add(int(e['target_id']))
        for n, v in nodes.items():
            if any(f in v for f in PF_N): flagged.add(n); synth.add(n)
        for ci, c in enumerate(comps):
            if csize[ci] >= MINLEN or fork[ci]: continue
            if padd_mode == 'any': hit = any(u in flagged for u in c)
            elif padd_mode == 'synth': hit = all(u in synth for u in c)            # made only of P-stage synthetic nodes
            else: hit = bool(ce.get(ci)) and all(ce[ci])                            # 'alledge': every edge is a P-stage addition
            if hit:
                new = set(c) - drop
                if new: st['pa_comp'] += 1; st['pa_nodes'] += len(new)
                drop.update(c)
    return drop, dict(st)


def prune(nodes, edges, drop):
    if not drop: return nodes, edges
    return ({k: v for k, v in nodes.items() if k not in drop},
            [e for e in edges if int(e['source_id']) not in drop and int(e['target_id']) not in drop])


def apply(nodes, edges, name=None, zarr=None, **kw):
    kw = {k: v for k, v in kw.items() if k not in ('set', 'fullgeff')}
    shape = _shape(zarr) if zarr else None
    drop, st = select(nodes, edges, shape=shape, **kw)
    st['removed_nodes'] = len(drop)
    nn, ne = prune(nodes, edges, drop)
    return nn, ne, st
