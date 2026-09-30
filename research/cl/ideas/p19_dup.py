"""P19 family 'dup' (GT-free, deletion only; no edge is ever created). Applied to the final P17 graph.
  start : START-side mirror of P14 term_trim. A track START n (no parent, exactly one child, not a B5-reference fork daughter,
          forward linear length >= minlen) lying within r um (rounded submission coords, as term_trim) of a same-frame node m that
          continues THROUGH the frame (m has a parent and a child) is a duplicate head: delete n. Iterated up to iters times, so the
          duplicated head is removed up to where the two tracks separate. (m = END is the term_trim join case: left alone here.)
  tt    : P14 term_trim re-applied unchanged on the final graph (coverage check: relinefit/tbext/P17 run after term_trim).
  par   : two linear tracks whose nodes stay within r um (rounded coords) of each other for >= minrun consecutive frames (same cell
          detected twice). Delete the overlapping nodes of the shorter (or newer) segment, only if the overlap reaches that
          segment's head or tail (deletion leaves one piece, never two fragments). Sister pairs (same fork parent) are left alone.
"""
from collections import defaultdict
import numpy as np

WANTS_META = True
S = np.array([1.625, 0.40625, 0.40625])
REF = {'hold36': '/workspace/runs/b5f_hold36', 'prev4': '/workspace/runs/b5f_prev4', 'audit32': '/workspace/sync3/runs/b5f_audit32',
       't127a': '/workspace/sync4/runs/b5f_t127a', 't127b': '/workspace/sync3/runs/b5f_t127b'}


def _struct(edges):
    out = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
    return out, par


def _rpos(nodes):
    return {n: np.array([max(0, int(round(float(v[k])))) for k in 'zyx']) * S for n, v in nodes.items()}


def _drop(nodes, edges, rm):
    return ({k: v for k, v in nodes.items() if k not in rm},
            [e for e in edges if int(e['source_id']) not in rm and int(e['target_id']) not in rm])


def ref_fork_daughters(refp):
    import json
    d = json.loads(open(refp).read()); ro = defaultdict(list)
    for e in d['edges']: ro[int(e['source_id'])].append(int(e['target_id']))
    return {c for s, cs in ro.items() if len(cs) >= 2 for c in cs}


def start_trim(nodes, edges, D=frozenset(), r=3.5, minlen=3, iters=5):
    from scipy.spatial import cKDTree
    total = 0
    for it in range(iters):
        out, par = _struct(edges); pos = _rpos(nodes)
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
                if n in par or len(out.get(n, [])) != 1 or n in D: continue  # only plain track starts
                if tlen_fwd(n) < minlen: continue
                for j in tr.query_ball_point(pos[n], r):
                    m = ns[j]
                    if m == n or m in drop or m not in par or not out.get(m): continue
                    drop.add(n); break
        if not drop: break
        total += len(drop)
        nodes, edges = _drop(nodes, edges, drop)
    return nodes, edges, total


def par_dup(nodes, edges, r=2.5, minrun=3, which='shorter'):
    from scipy.spatial import cKDTree
    out, par = _struct(edges); pos = _rpos(nodes)
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    close = set()
    for t, ns in byt.items():
        if len(ns) < 2: continue
        tr = cKDTree(np.stack([pos[n] for n in ns]))
        for i, j in tr.query_pairs(r):
            a, b = ns[i], ns[j]; close.add((a, b)); close.add((b, a))

    def s1(n):  # unique linear successor
        c = out.get(n, [])
        return c[0] if len(c) == 1 else None

    def seg(n):  # linear segment containing n: back while parent has one child, forward while one child
        h = n
        while h in par and len(out[par[h]]) == 1: h = par[h]
        L = [h]
        while s1(L[-1]) is not None: L.append(s1(L[-1]))
        return L
    segc = {}
    rm = set(); nrun = 0
    for (a, b) in sorted(close):
        if a > b: continue
        pa, pb = par.get(a), par.get(b)
        if pa is not None and pb is not None and (pa, pb) in close and s1(pa) == a and s1(pb) == b: continue  # not a run start
        if pa is not None and pa == pb: continue  # sisters
        run = [(a, b)]
        while True:
            x, y = s1(run[-1][0]), s1(run[-1][1])
            if x is None or y is None or (x, y) not in close: break
            run.append((x, y))
        if len(run) < minrun: continue
        nrun += 1
        sa = segc.get(a) or seg(a); sb = segc.get(b) or seg(b)
        for n in sa: segc[n] = sa
        for n in sb: segc[n] = sb
        ta0, tb0 = int(nodes[sa[0]]['t']), int(nodes[sb[0]]['t'])
        if which == 'shorter': key = lambda k: ((len(sa), -ta0, a) if k == 0 else (len(sb), -tb0, b))
        else: key = lambda k: ((-ta0, len(sa), a) if k == 0 else (-tb0, len(sb), b))
        order = sorted([0, 1], key=key)
        k = order[0]  # the shorter / newer track
        sg = sa if k == 0 else sb
        ov = [p[k] for p in run]
        if ov[0] == sg[0] or ov[-1] == sg[-1]: rm.update(ov)  # overlap interior to the chosen segment: leave it
    if not rm: return nodes, edges, 0, nrun
    nodes, edges = _drop(nodes, edges, rm)
    return nodes, edges, len(rm), nrun


def apply(nodes, edges, mode='start', r=3.5, minlen=3, iters=5, excl_fd=1, which='shorter', minrun=3, pr=2.5, tt_join='keep_end', name=None, set=None, **kw):
    n0 = len(nodes); st = {}
    for md in mode.split('+'):
        if md == 'start':
            D = ref_fork_daughters(REF[set] + '/working/reference_graphs/' + name + '.json') if excl_fd else frozenset()
            nodes, edges, k = start_trim(nodes, edges, D, r=r, minlen=minlen, iters=iters); st['st_rm'] = k
        elif md == 'tt':
            import sys
            if '/workspace/p12ds' not in sys.path: sys.path.insert(0, '/workspace/p12ds')
            import p14_post
            nodes, edges, s2 = p14_post.term_trim(nodes, edges, join=tt_join); st['tt_rm'] = s2['trim']
        elif md == 'par':
            nodes, edges, k, nr = par_dup(nodes, edges, r=pr, minrun=minrun, which=which); st['par_rm'] = k; st['par_runs'] = nr
        else:
            raise ValueError(md)
    st['dup_dropped'] = n0 - len(nodes)
    return nodes, edges, st


def post(nodes, edges, refp, start_r=2.5, par_r=3.5, tt_join=None):
    """Deployable P-stage post-step (after p17_post): pure deletion, no edge created when tt_join is None.
    refp = the B5 reference graph (<run>/working/reference_graphs/<movie>.json), used only for its fork daughters."""
    import sys
    if '/workspace/p12ds' not in sys.path: sys.path.insert(0, '/workspace/p12ds')
    import p14_post
    n0 = len(nodes)
    nodes, edges, k1 = start_trim(nodes, edges, ref_fork_daughters(refp), r=start_r, minlen=3, iters=5)
    nodes, edges, s2 = p14_post.term_trim(nodes, edges, join=tt_join)
    nodes, edges, k3, nr = par_dup(nodes, edges, r=par_r, minrun=3, which='shorter')
    return nodes, edges, {'p19_st': k1, 'p19_tt': s2['trim'], 'p19_par': k3, 'p19_dup': n0 - len(nodes)}