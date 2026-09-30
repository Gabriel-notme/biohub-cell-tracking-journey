"""Is the dense transformer probability P(parent=p | b) informative for division-completion candidates (p->a, second daughter b)?"""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
ART = '/workspace/art_b56/artifact_bundle'; os.environ['BIOHUB_ART'] = ART
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, ART); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np


def job(name):
    import evalx, dense_relink, div_complete as dc
    from tracking_cellmot.metrics import evaluate
    from tracking_cellmot.division_metrics import score_divisions
    K = evalx.K
    nodes, edges = evalx.load_graph_json(Path('/workspace/runs/b5d_train/working/lineage_graphs') / (name + '.json'))
    rows, out, prev, pos = dc.candidates(nodes, edges)
    dp = dense_relink.load_dense(Path('/workspace/runs/dense_train') / (name + '.npz'), nodes)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    recovered = {int(d) for d, v in res.scores.items() if v}
    pred2, mapping2 = evalx.to_graph(nodes, edges); inv2 = {v: k for k, v in mapping2.items()}
    evaluate(pred2, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred2.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv2[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    ea = gt.edge_attrs(); gs = defaultdict(list); gp = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(x)].append(int(y)); gp[int(y)] = int(x)
    out_rows = []
    for p, a, b, q, typ in rows:
        g_p, g_a, g_b = p2g.get(p), p2g.get(a), p2g.get(b)
        lab = 'U'
        if g_p is not None and gs.get(g_p):
            lab = 'N'
            for gd in [g_p] + ([gp[g_p]] if g_p in gp else []) + list(gs.get(g_p, [])):
                ch = gs.get(gd, [])
                if len(ch) != 2: continue
                lin = [set([x] + gs.get(x, [])) for x in ch]
                if g_a is not None and g_b is not None and any(g_a in L for L in lin) and any(g_b in L for L in lin) and not any(g_a in L and g_b in L for L in lin):
                    lab = 'D' if gd in recovered else 'P'; break
        if lab == 'U': continue
        out_rows.append((typ, lab, dp.get((p, b), 0.), dp.get((q, b), 0.) if q is not None else -1., dp.get((p, a), 0.)))
    return out_rows


if __name__ == '__main__':
    names = [l.strip() for l in open('/workspace/cl/dense_train.txt') if l.strip()]
    with Pool(32) as pool: R = [x for xs in pool.map(job, names) for x in xs]
    def auc(pp, yy):
        pp = np.asarray(pp); yy = np.asarray(yy); o = np.argsort(pp); rr = np.empty(len(pp)); rr[o] = np.arange(len(pp)); n1 = yy.sum(); n0 = len(yy) - n1
        return (rr[yy == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)
    for typ in ['start', 'stolen']:
        q = [r for r in R if r[0] == typ]
        y = np.array([r[1] == 'P' for r in q], int)
        print(typ, 'n', len(q), Counter(r[1] for r in q))
        if y.sum() == 0: continue
        dpb = np.array([r[2] for r in q]); dqb = np.array([r[3] for r in q])
        print('   AUC dp(p,b) %.3f  dp(p,b)-dp(q,b) %.3f' % (auc(dpb, y), auc(dpb - np.maximum(dqb, 0), y)))
        for th in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
            m = dpb >= th; print('   dp(p,b)>=%.2f: P %d D %d N %d' % (th, (m & (y == 1)).sum(), sum(1 for r, mm in zip(q, m) if mm and r[1] == 'D'), sum(1 for r, mm in zip(q, m) if mm and r[1] == 'N')))
