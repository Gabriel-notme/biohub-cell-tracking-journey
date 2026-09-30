"""Round-2 FREE-END linking checks on top of P14 (= P13 + ideas.combo14 when base='p13'). No GT.
mode none : P14 itself
mode ll   : a second long_link pass with other (fe_min, min_um)  (pre-ILP end->start candidate edges, greedy by prob)
mode mnn  : mutual-nearest end(t)->start(t+1) within R um where the start is the nearest node of ANY kind at t+1 and the end is
            the nearest END at t of that start; both tracks >= minlen nodes."""
import numpy as np
WANTS_META = True
S = np.array([1.625, .40625, .40625])


def apply(nodes, edges, mode='none', base='p13', fe_min=0.5, min_um=14.0, R=4.0, minlen=3, name=None, set=None, fullgeff=None, zarr=None):
    import sys
    sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/cl/ideas')
    from ideas import combo14
    st = {}
    if base == 'p13':
        nodes, edges, st = combo14.apply(nodes, edges, fullgeff=fullgeff)
    if mode == 'none':
        return nodes, edges, st
    if mode == 'll':
        import pp_longlink
        nodes, edges, s = pp_longlink.apply(nodes, edges, fullgeff=fullgeff, fe_min=fe_min, min_um=min_um)
        st.update(s); return nodes, edges, st
    from collections import defaultdict
    from scipy.spatial import cKDTree
    succ = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a
    pos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx']) * S for n, v in nodes.items()}
    T = {n: int(v['t']) for n, v in nodes.items()}

    def blen(n):
        c = 1
        while n in par: n = par[n]; c += 1
        return c

    def flen(n):
        c = 1
        while len(succ.get(n, [])) == 1: n = succ[n][0]; c += 1
        return c
    by = defaultdict(list); ends = defaultdict(list)
    for n in nodes:
        by[T[n]].append(n)
        if not succ.get(n) and n in par and len(succ.get(par[n], [])) == 1: ends[T[n]].append(n)
    atree = {t: (ns, cKDTree(np.stack([pos[n] for n in ns]))) for t, ns in by.items()}
    etree = {t: (ns, cKDTree(np.stack([pos[n] for n in ns]))) for t, ns in ends.items()}
    add = []
    for t, ens in ends.items():
        if t + 1 not in atree: continue
        ns1, tr1 = atree[t + 1]; nse, tre = etree[t]
        for a in ens:
            d, j = tr1.query(pos[a], k=1)
            b = ns1[int(j)]
            if d > R or b in par or not succ.get(b): continue
            d2, j2 = tre.query(pos[b], k=1)
            if nse[int(j2)] != a: continue
            if blen(a) < minlen or flen(b) < minlen: continue
            add.append({'source_id': a, 'target_id': b, 'mnn_link': round(float(d), 2)})
    st['mnn_added'] = len(add)
    return nodes, list(edges) + add, st
