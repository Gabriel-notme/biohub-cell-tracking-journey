"""check2/reimpl: independent reimplementation of the P19 'dup' and 'border' deletion rules from their written description only
(task text + p19r.py / p19small.py docstrings + public signatures of p19_dup / p19_edge_deploy; their bodies were NOT read).

dup    = (1) START duplicate heads within start_r (mirror of P14 term_trim: a track START with a child lying within start_r um of a
             same-frame node that has a parent is a duplicate head; delete it, repeat up to iters so the whole duplicated head goes;
             fork parents and B5-reference fork daughters are never touched; the trimmed track must have >= minlen nodes)
         (2) P14 term_trim with join=None (called from the deployed p14_post.py, as the description says 'P14 term_trim')
         (3) parallel duplicate overlap at par_r (my reading: a fork-free track whose frames all lie within par_r of one other track,
             with >= minrun common frames, is a duplicate; delete the shorter of the two)
border = weakly connected, fork-free components with < minlen nodes whose every node lies within `margin` voxels of the lateral
         (y/x) FOV border.
"""
import sys, json
from collections import defaultdict
import numpy as np
sys.path.insert(0, '/workspace/p19ds')
S = np.array([1.625, .40625, .40625])


def _struct(edges):
    out = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
    return out, par


def _pos(nodes, rounded):
    if rounded:
        return {n: np.array([max(0, int(round(float(v[k])))) for k in 'zyx'], float) * S for n, v in nodes.items()}
    return {n: np.array([float(v[k]) for k in 'zyx']) * S for n, v in nodes.items()}


def _drop(nodes, edges, drop):
    return ({k: v for k, v in nodes.items() if k not in drop},
            [e for e in edges if int(e['source_id']) not in drop and int(e['target_id']) not in drop])


def ref_daughters(refp):
    d = json.loads(open(refp).read()); ro = defaultdict(list)
    for e in d['edges']: ro[int(e['source_id'])].append(int(e['target_id']))
    return {c for s, cs in ro.items() if len(cs) >= 2 for c in cs}


def start_heads(nodes, edges, D=frozenset(), r=2.5, minlen=3, iters=5, rounded=True, need_par=True, need_child=False):
    from scipy.spatial import cKDTree
    total = 0
    for it in range(iters):
        out, par = _struct(edges); pos = _pos(nodes, rounded)
        byt = defaultdict(list)
        for n, v in nodes.items(): byt[int(v['t'])].append(n)

        def tlen_fwd(n):
            L = 1
            while len(out.get(n, [])) == 1: n = out[n][0]; L += 1
            return L
        drop = set()
        for t, ns in byt.items():
            if len(ns) < 2: continue
            tr = cKDTree(np.stack([pos[n] for n in ns]))
            for n in ns:
                if n in par or not out.get(n): continue      # only track starts that continue
                if len(out[n]) >= 2 or n in D: continue       # fork parent / B5 fork daughter
                if tlen_fwd(n) < minlen: continue
                for j in tr.query_ball_point(pos[n], r):
                    m = ns[j]
                    if m == n or m in drop: continue
                    if need_par and m not in par: continue
                    if need_child and not out.get(m): continue
                    drop.add(n); break
        if not drop: break
        total += len(drop); nodes, edges = _drop(nodes, edges, drop)
    return nodes, edges, total


def parallel(nodes, edges, r=3.5, minrun=3, rounded=True):
    from scipy.spatial import cKDTree
    out, par = _struct(edges); pos = _pos(nodes, rounded)
    adj = defaultdict(list)
    for a, cs in out.items():
        for b in cs: adj[a].append(b); adj[b].append(a)
    comp = {}; comps = []
    for n in nodes:
        if n in comp: continue
        c = []; st = [n]; comp[n] = len(comps)
        while st:
            u = st.pop(); c.append(u)
            for v in adj.get(u, []):
                if v not in comp: comp[v] = len(comps); st.append(v)
        comps.append(c)
    forkfree = [not any(len(out.get(u, [])) >= 2 for u in c) for c in comps]
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    near = defaultdict(lambda: defaultdict(set))   # comp -> partner comp -> frames within r
    for t, ns in byt.items():
        if len(ns) < 2: continue
        tr = cKDTree(np.stack([pos[n] for n in ns]))
        for i, j in tr.query_pairs(r):
            a, b = comp[ns[i]], comp[ns[j]]
            if a == b: continue
            near[a][b].add(t); near[b][a].add(t)
    dead = set(); rm = set()
    order = sorted(range(len(comps)), key=lambda k: (len(comps[k]), min(comps[k])))
    for k in order:
        if not forkfree[k] or len(comps[k]) < minrun: continue
        T = {int(nodes[u]['t']) for u in comps[k]}
        for p, fr in near[k].items():
            if p in dead or len(comps[p]) < len(comps[k]): continue
            if T <= fr:
                dead.add(k); rm.update(comps[k]); break
    if not rm: return nodes, edges, 0
    n2, e2 = _drop(nodes, edges, rm)
    return n2, e2, len(rm)


def parallel2(nodes, edges, r=3.5, minrun=3, rounded=True, which='shorter', seglen='seg'):
    """v2 (after black-box diff showed head/tail runs): a START head run (END tail run) of a fork-free segment A whose first (last)
    >= minrun consecutive frames each lie within r of the same other segment B is a duplicate overlap; delete that run from A when A
    is the shorter of the two (ties: A has the larger first id), segments processed shortest first."""
    from scipy.spatial import cKDTree
    out, par = _struct(edges); pos = _pos(nodes, rounded)
    seg_of = {}; segs = []
    for n in nodes:
        if n in par and len(out.get(par[n], [])) == 1: continue
        s = [n]; c = n
        while len(out.get(c, [])) == 1: c = out[c][0]; s.append(c)
        for u in s: seg_of[u] = len(segs)
        segs.append(s)
    T = {n: int(v['t']) for n, v in nodes.items()}
    byt = defaultdict(list)
    for n in nodes: byt[T[n]].append(n)
    fr = defaultdict(lambda: defaultdict(set))
    for t, ns in byt.items():
        if len(ns) < 2: continue
        tr = cKDTree(np.stack([pos[n] for n in ns]))
        for i, j in tr.query_pairs(r):
            a, b = seg_of[ns[i]], seg_of[ns[j]]
            if a != b: fr[a][b].add(t); fr[b][a].add(t)
    rm = set(); gone = set()
    order = sorted(range(len(segs)), key=lambda k: (len(segs[k]), segs[k][0]))
    for k in order:
        A = segs[k]
        for side in ('head', 'tail'):
            if side == 'head' and A[0] in par: continue
            if side == 'tail' and out.get(A[-1]): continue
            seq = A if side == 'head' else A[::-1]
            best = []
            for b, F in fr[k].items():
                B = segs[b]
                if which == 'shorter' and (len(B) < len(A) or (len(B) == len(A) and B[0] > A[0])): continue
                run = []
                for u in seq:
                    if u in rm or T[u] not in F: break
                    # the partner node at that frame must still exist
                    if not any(T[w] == T[u] and w not in rm for w in B): break
                    run.append(u)
                if len(run) > len(best): best = run
            if len(best) >= minrun: rm.update(best)
    if not rm: return nodes, edges, 0
    n2, e2 = _drop(nodes, edges, rm)
    return n2, e2, len(rm)


def dup(nodes, edges, refp, start_r=2.5, par_r=3.5, order='st', m_child=False, par='v2', which='shorter'):
    import p14_post
    D = ref_daughters(refp)
    st = {}
    for step in order:
        if step == 's':
            nodes, edges, a = start_heads(nodes, edges, D=D, r=start_r, need_child=m_child); st['m_sh'] = a
        else:
            nodes, edges, s = p14_post.term_trim(nodes, edges, join=None); st['m_tt'] = s['trim']
    if par == 'none':
        c = 0
    elif par == 'v1':
        nodes, edges, c = parallel(nodes, edges, r=par_r)
    else:
        nodes, edges, c = parallel2(nodes, edges, r=par_r, which=which)
    st['m_par'] = c
    return nodes, edges, st


def border(nodes, edges, shape_yx=(256, 256), minlen=6, margin=2.0, hi_off=1, strict=False):
    out, par = _struct(edges)
    adj = defaultdict(list)
    for a, cs in out.items():
        for b in cs: adj[a].append(b); adj[b].append(a)

    def nb(v):
        y, x = float(v['y']), float(v['x'])
        H, W = shape_yx[0] - hi_off, shape_yx[1] - hi_off
        d = min(y, x, H - y, W - x)
        return d < margin if strict else d <= margin
    seen = set(); rm = set()
    for n in nodes:
        if n in seen: continue
        c = []; st = [n]; seen.add(n)
        while st:
            u = st.pop(); c.append(u)
            for v in adj.get(u, []):
                if v not in seen: seen.add(v); st.append(v)
        if len(c) >= minlen or any(len(out.get(u, [])) >= 2 for u in c): continue
        if all(nb(nodes[u]) for u in c): rm.update(c)
    if not rm: return nodes, edges, 0
    n2, e2 = _drop(nodes, edges, rm)
    return n2, e2, len(rm)


B5 = {'hold36': '/workspace/runs/b5f_hold36', 'prev4': '/workspace/runs/b5f_prev4', 'audit32': '/workspace/sync3/runs/b5f_audit32',
      't127a': '/workspace/sync4/runs/b5f_t127a', 't127b': '/workspace/sync3/runs/b5f_t127b'}
WANTS_META = True


def apply(nodes, edges, start_r=2.5, par_r=3.5, margin=2.0, bminlen=6, name=None, set=None, dup_kw=None, border_kw=None, border_on=1,
          p17_on=1, **kw):
    """default dup_kw below = the variant that reproduces p19r.py exactly on all 199 P15 graphs."""
    sys.path.insert(0, '/workspace/cl/p16/deploy')
    import p17_post
    refp = B5[set] + '/working/reference_graphs/%s.json' % name
    a = b = 0
    if p17_on:
        nodes, edges, a = p17_post.cutdup(nodes, edges)
        nodes, edges, b = p17_post.forkfrag(nodes, edges, refp)
    dk = dict({'order': 'st', 'm_child': True, 'par': 'v2'}, **(dup_kw or {}))
    nodes, edges, s = dup(nodes, edges, refp, start_r=start_r, par_r=par_r, **dk)
    c = 0
    if border_on:
        nodes, edges, c = border(nodes, edges, minlen=bminlen, margin=margin, **(border_kw or {}))
    return nodes, edges, dict(s, cd=a, ff=b, m_border=c)

