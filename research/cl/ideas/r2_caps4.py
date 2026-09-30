"""r2_caps4 (diagnostic, reads GT): relink skip-rule 'd is a fork daughter' on P14: ILP edges s->d (s = track END, d = daughter of fork q).
Per candidate: ILP prob, s->d label, q->d label, fork q official label (tp/fp/nc), fork origin (div_complete or not), and whether s was q's
old parent (div_complete stolen). Also E->S/E->T1 structural FN: |s - thief q| and thief history. Writes r2_caps4_out.json"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
for p in ['/workspace/cl', '/workspace/cl/ideas', '/workspace/official/src', '/workspace/code', '/workspace/p56stage']:
    sys.path.insert(0, p)
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}


def job(f):
    try:
        import numcodecs.blosc
        numcodecs.blosc.use_threads = False
    except Exception:
        pass
    import evalx, combo14, edge_link
    import tracksdata as td
    from tracking_cellmot import metrics as M
    from tracking_cellmot.division_metrics import score_divisions
    K = td.DEFAULT_ATTR_KEYS
    name = Path(f).stem; st = f.split('ps_p13_')[1].split('/')[0]; emb = name[:4]
    fg = FULL[st] + '/' + name + '.geff'
    nodes, edges = evalx.load_graph_json(f)
    nodes, edges, _ = combo14.apply(nodes, edges, fullgeff=fg)
    fids, fT, fV, fE, fprob = edge_link.load_full(fg)
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    M._evaluate(pred, gt, 'jaccard', evalx.SCALE, 7.)
    ea = M._evaluate_matched_graph(pred, gt)
    lab = {}
    for s, d, m, v in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list(), ea[K.MATCHED_EDGE_MASK].to_list(), ea['pred_valid'].to_list()):
        lab[(inv[s], inv[d])] = 'TP' if m else ('FP' if v else 'U')
    pred2, mapping2 = evalx.to_graph(nodes, edges); inv2 = {v: k for k, v in mapping2.items()}
    res = score_divisions(pred2, gt, scale=evalx.SCALE, max_distance=7.)
    tpf = {inv2[int(x)] for x in res.tp_forks}; fpf = {inv2[int(x)] for x in res.fp_forks}
    out = defaultdict(list); par = {}; isdc = set()
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
        if 'div_complete' in e: isdc.add(a)
    rows = []
    for (a, b), p in zip(fE.tolist(), fprob.tolist()):
        a, b = int(a), int(b)
        if a not in nodes or b not in nodes or out.get(a) or b not in par: continue
        q = par[b]
        if q == a or len(out[q]) < 2: continue
        rows.append(dict(m=name, emb=emb, fe=float(p), fork='tp' if q in tpf else ('fp' if q in fpf else 'nc'), dc=q in isdc,
                         s_is_fdau=(a in par and len(out[par[a]]) >= 2), qd=lab[(q, b)]))
    return rows


if __name__ == '__main__':
    files = sorted(glob.glob('/workspace/cl/ps_p13_*/graphs/*.json'))
    with Pool(40, maxtasksperchild=4) as p: R = [x for rs in p.map(job, files, chunksize=1) for x in rs]
    json.dump(R, open('/workspace/cl/ideas/r2_caps4_out.json', 'w'))
    c = Counter()
    for x in R:
        c[(x['emb'], 'fe>=.9' if x['fe'] >= .9 else ('fe>=.5' if x['fe'] >= .5 else 'fe<.5'), 'dc' if x['dc'] else 'b5', x['fork'], 'qd' + x['qd'])] += 1
    for k in sorted(c): print(k, c[k])
