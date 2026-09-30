"""dv_oracle (ORACLE, reads GT): rewire every FN GT division of the chosen categories into a fork (P -> unlinked daughter Q; Q's old
incoming edge removed). Categories: start (unlinked daughter's track starts at t+1/t+2), early (stolen, q-track started <=3 frames
before t), long (stolen, q-track older). Optional fp_rate: also add that many random wrong forks per true one (evaluable p only)
to emulate a classifier with given precision. Used through rule_eval.py p15 nm.dv_oracle '[{"cats": [...]}]'."""
import sys
from collections import defaultdict
import numpy as np
WANTS_META = True
S = np.array([1.625, 0.40625, 0.40625])


def apply(nodes, edges, cats=('start', 'early', 'long'), emb=None, name=None, **kw):
    import evalx
    from tracking_cellmot.division_metrics import score_divisions, _match_full, _matched_node_attrs
    if emb and not name.startswith(emb): return nodes, edges, {'or_added': 0}
    K = evalx.K
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(x)]: int(y) for x, y in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    g2p = {y: x for x, y in p2g.items()}
    na = gt.node_attrs(attr_keys=[K.NODE_ID, 't'])
    gt_t = {int(i): int(t) for i, t in zip(na[K.NODE_ID].to_list(), na['t'].to_list())}
    ea = gt.edge_attrs(); gch = defaultdict(list); gpar = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gch[int(x)].append(int(y)); gpar[int(y)] = int(x)
    out = defaultdict(list); par = {}
    for e in edges:
        x, y = int(e['source_id']), int(e['target_id']); out[x].append(y); par[y] = x
    tt = {n: int(v['t']) for n, v in nodes.items()}

    def anc(n, k):
        r = []
        while n in par and len(r) < k: n = par[n]; r.append(n)
        return r

    def root(n):
        k = 0
        while n in par and k < 300: n = par[n]; k += 1
        return n
    rm, add = set(), []
    st = defaultdict(int)
    for d, v in res.scores.items():
        d = int(d)
        if v: continue
        P = g2p.get(d)
        if P is None: continue
        Q = []
        for c in gch[d][:2]:
            q = g2p.get(c)
            if q is None: q = next((g2p[g] for g in gch.get(c, []) if g in g2p), None)
            Q.append(q)
        if any(q is None for q in Q): continue
        linked = [P in anc(q, 3) for q in Q]
        if sum(linked) != 1: continue
        q = Q[linked.index(False)]
        rt = root(q)
        cat = 'start' if tt[rt] >= gt_t[d] + 1 else ('early' if tt[rt] >= gt_t[d] - 2 else 'long')
        if cat not in cats: continue
        if len(out.get(P, [])) != 1: continue
        # fork node: P if q is at t+1, else P's child at t+1
        F = P
        if tt[q] == tt[P] + 2:
            F = out[P][0]
            if tt[F] != tt[P] + 1 or len(out.get(F, [])) != 1: continue
        elif tt[q] != tt[P] + 1: continue
        if q in par: rm.add((par[q], q))
        add.append((F, q)); st['or_' + cat] += 1
    ne = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in rm] + [{'source_id': a, 'target_id': b, 'oracle': 1} for a, b in add]
    st['or_added'] = len(add)
    return nodes, ne, dict(st)
