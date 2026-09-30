"""dv_fn_tax (diagnostic, reads GT): taxonomy of every GT division on P15 graphs (TP/FN) and census of daughter-birth candidates.
For each FN: parent/daughter matched? which daughter is linked to the parent track, what the unlinked daughter's track looks like
(start at t+1 / later start / stolen from matched cell / stolen from unmatched duplicate), distances.
Also: all 'birth' candidates = (p continuing at t, s track start at t+1 within 13 um) and whether evaluable / positive.
Writes /workspace/cl/nm/dv_fn_tax_rows.json"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
for p in ['/workspace/cl', '/workspace/official/src', '/workspace/code']:
    sys.path.insert(0, p)
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']


def job(a):
    s, name = a
    import evalx
    from tracking_cellmot.division_metrics import score_divisions, _match_full, _matched_node_attrs
    K = evalx.K
    nodes, edges = evalx.load_graph_json('/workspace/cl/ps_p15_%s/graphs/%s.json' % (s, name))
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(x)]: int(y) for x, y in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    g2p = {y: x for x, y in p2g.items()}
    na = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x]) * S for i, z, y, x in zip(na[K.NODE_ID].to_list(), na['z'].to_list(), na['y'].to_list(), na['x'].to_list())}
    gt_t = {int(i): int(t) for i, t in zip(na[K.NODE_ID].to_list(), na['t'].to_list())}
    ea = gt.edge_attrs(); gch = defaultdict(list); gpar = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gch[int(x)].append(int(y)); gpar[int(y)] = int(x)
    out = defaultdict(list); par = {}
    for e in edges:
        x, y = int(e['source_id']), int(e['target_id']); out[x].append(y); par[y] = x
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
    tt = {n: int(v['t']) for n, v in nodes.items()}

    def root(n):
        k = 0
        while n in par and k < 300: n = par[n]; k += 1
        return n, k

    def anc(n, k):  # ancestors up to k steps
        r = []
        while n in par and len(r) < k: n = par[n]; r.append(n)
        return r
    divs = []
    for d, v in res.scores.items():
        d = int(d); ch = gch[d]
        r = dict(movie=name, set=s, emb=name[:4], gd=d, t=gt_t[d], tp=int(v))
        if v: divs.append(r); continue
        P = g2p.get(d)
        if P is None and d in gpar and gpar[d] in g2p:
            gp_ = g2p[gpar[d]]; P = next((c for c in out.get(gp_, []) if tt[c] == gt_t[d]), None) or gp_
            r['pflag'] = 'via_gp'
        if P is None:
            r['cat'] = 'parent_miss'
            # nearest pred node at t to the GT parent
            dd = [float(np.linalg.norm(pos[n] - gpos[d])) for n in nodes if tt[n] == gt_t[d]]
            r['near_parent'] = min(dd) if dd else -1
            divs.append(r); continue
        Q = []
        for c in ch[:2]:
            q = g2p.get(c)
            if q is None:
                q = next((g2p[g] for g in gch.get(c, []) if g in g2p), None)
            Q.append(q)
        if any(q is None for q in Q):
            r['cat'] = 'daughter_miss'
            miss = [c for c, q in zip(ch, Q) if q is None][0]
            dd = [float(np.linalg.norm(pos[n] - gpos[miss])) for n in nodes if tt[n] == gt_t[miss]]
            r['near_daughter'] = min(dd) if dd else -1
            divs.append(r); continue
        linked = [P in anc(q, 3) or any(x in anc(q, 3) for x in anc(P, 1)) for q in Q]
        r['linked'] = linked
        r['d_PQ'] = [float(np.linalg.norm(pos[P] - pos[q])) for q in Q]
        r['d_QQ'] = float(np.linalg.norm(pos[Q[0]] - pos[Q[1]]))
        r['nfork_P'] = len(out.get(P, []))
        if all(linked):
            r['cat'] = 'both_linked'  # fork exists somewhere but rejected
        elif not any(linked):
            r['cat'] = 'none_linked'
            r['P_out'] = len(out.get(P, [])); r['P_end'] = int(len(out.get(P, [])) == 0)
        else:
            q = Q[linked.index(False)]
            rt, k = root(q)
            r['q_track_start_dt'] = tt[rt] - gt_t[d]  # start time of the unlinked daughter's track relative to division
            if tt[rt] >= gt_t[d] + 1:
                r['cat'] = 'start'
                r['start_node_d_P'] = float(np.linalg.norm(pos[rt] - pos[P])) if tt[rt] == gt_t[d] + 1 else -1
            else:
                # stolen: what is q's track before the division
                pre = [x for x in anc(q, 60) if tt[x] <= gt_t[d]]
                m = [x for x in pre if x in p2g]
                r['cat'] = 'stolen_real' if len(m) >= 0.5 * max(1, len(pre)) else 'stolen_dup'
                r['stolen_pre_len'] = len(pre); r['stolen_pre_matched'] = len(m)
                x_t = next((x for x in pre if tt[x] == gt_t[d]), None)
                r['stolen_dQP_t'] = float(np.linalg.norm(pos[x_t] - pos[P])) if x_t is not None else -1
        divs.append(r)
    # birth candidates census: p with exactly one child at t, s = track start at t+1 within 13 um of p
    frames = defaultdict(list)
    for n in nodes: frames[tt[n]].append(n)
    starts = [n for n in nodes if n not in par and tt[n] > 0]
    fk = {n for n in nodes if len(out.get(n, [])) >= 2}
    posdiv = set()
    for d in gch:
        if len(gch[d]) >= 2: posdiv.add(d)
    births = dict(n=0, evaluable=0, pos=0, pos_new=0)
    tpfk = {inv[int(x)] for x in res.tp_forks}
    rec = {int(d) for d, v in res.scores.items() if v}
    from scipy.spatial import cKDTree
    for t, fr in frames.items():
        st = [n for n in starts if tt[n] == t + 1]
        if not st: continue
        arr = np.array([pos[n] for n in fr]); tree = cKDTree(arr)
        for sn in st:
            for j in tree.query_ball_point(pos[sn], 13.0):
                p = fr[j]
                if len(out.get(p, [])) != 1: continue
                births['n'] += 1
                g = p2g.get(p)
                if g is not None and gch.get(g):
                    births['evaluable'] += 1
                    gs_ = p2g.get(sn)
                    for gd in [g] + ([gpar[g]] if g in gpar else []):
                        if gd in posdiv and gs_ is not None and any(gs_ == c or gs_ in gch.get(c, []) for c in gch[gd]):
                            births['pos'] += 1; births['pos_new'] += int(gd not in rec); break
    return dict(movie=name, set=s, divs=divs, births=births, n_forks=len(fk), n_starts=len(starts), n_nodes=len(nodes), n_edges=len(edges))


if __name__ == '__main__':
    jobs = [(s, Path(f).stem) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s))]
    with Pool(24, maxtasksperchild=4) as p: R = p.map(job, jobs, chunksize=1)
    json.dump(R, open('/workspace/cl/nm/dv_fn_tax_rows.json', 'w'))
    from collections import Counter
    c = Counter()
    for r in R:
        for d in r['divs']:
            c[(d['emb'], 'TP' if d['tp'] else d['cat'])] += 1
    for k in sorted(c): print(k, c[k])
    b = Counter()
    for r in R:
        for k, v in r['births'].items(): b[(r['movie'][:4], k)] += v
        b[(r['movie'][:4], 'forks')] += r['n_forks']; b[(r['movie'][:4], 'starts')] += r['n_starts']; b[(r['movie'][:4], 'edges')] += r['n_edges']
    for k in sorted(b): print(k, b[k])
