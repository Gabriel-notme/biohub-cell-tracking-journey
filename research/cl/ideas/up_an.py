"""UPSTREAM lens (read-only analysis): relation of P13 edges / forks to the pre-ILP candidate graph (fullgraph).
Per P13 edge: candidate status in fullgraph (agree / b has other cand parent / b no cand parent / synthetic node), fullgraph prob,
official label TP / FP (evaluable) / U. Per P13 fork: candidate status of the two daughter edges, official fork label.
usage: python3 ideas/up_an.py  -> writes /workspace/cl/ideas/up_rows.json"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']


def etype(e):
    for k in ['gap_closed', 'edge_link', 'gap2_recovered', 'joint_event', 'relink', 'div_complete', 'safe_division']:
        if k in e: return k
    return e.get('motion_pass', 'other')


def job(a):
    s, f = a
    import evalx, zarr
    from tracking_cellmot.metrics import evaluate
    import tracking_cellmot.division_metrics as DM
    K = evalx.K
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    g = zarr.open_group(FULL[s] + '/' + name + '.geff', mode='r')
    fid = np.asarray(g['nodes/ids'][:]).astype(np.int64)
    fP = np.stack([np.asarray(g['nodes/props/%s/values' % k][:]) for k in 'tzyx'], 1)
    fE = np.asarray(g['edges/ids'][:]).astype(np.int64); fp = np.asarray(g['edges/props/edge_prob/values'][:])
    fidx = {int(i): j for j, i in enumerate(fid.tolist())}
    cpar = {}; cch = defaultdict(list); cprob = {}
    for (u, v), p in zip(fE.tolist(), fp.tolist()):
        cpar[v] = u; cch[u].append(v); cprob[(u, v)] = p
    # id consistency check
    dev = []
    for n, v in nodes.items():
        j = fidx.get(n)
        if j is not None and int(fP[j, 0]) == int(v['t']):
            dev.append(float(np.linalg.norm((fP[j, 1:] - np.array([v['z'], v['y'], v['x']])) * S)))
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    ea = gt.edge_attrs(); src = ea[K.EDGE_SOURCE].to_list(); dst = ea[K.EDGE_TARGET].to_list()
    ge = set(zip(src, dst)); gout = set(src); gin = set(dst)
    ch = defaultdict(list); par = {}
    for e in edges:
        u, v = int(e['source_id']), int(e['target_id']); ch[u].append(v); par[v] = u
    rows = []
    for e in edges:
        u, v = int(e['source_id']), int(e['target_id'])
        valid = (u in p2g and p2g[u] in gout) or (v in p2g and p2g[v] in gin)
        lab = 'U'
        if valid: lab = 'TP' if (u in p2g and v in p2g and (p2g[u], p2g[v]) in ge) else 'FP'
        if u not in fidx or v not in fidx: cat = 'synth'
        elif cpar.get(v) == u: cat = 'agree'
        elif v in cpar: cat = 'other_in_p13' if cpar[v] in nodes else 'other_dropped'
        else: cat = 'nocand'
        # does u have a candidate child that is in P13 and different from v?
        uc = [w for w in cch.get(u, []) if w != v]
        uc_in = [w for w in uc if w in nodes]
        uc_free = [w for w in uc_in if w not in par]
        cp = cpar.get(v); cp_in = cp is not None and cp in nodes
        cp_end = cp_in and len(ch.get(cp, [])) == 0
        rows.append({'lab': lab, 'cat': cat, 'et': etype(e), 'ep': e.get('edge_prob'), 'fp': cprob.get((u, v)),
                     'fp_alt_par': cprob.get((cp, v)) if cp is not None and cp != u else None,
                     'u_alt_ch_in': len(uc_in), 'u_alt_ch_free': len(uc_free), 'cp_in': int(bool(cp_in)), 'cp_end': int(bool(cp_end)),
                     'nch_u': len(ch[u]), 'd': float(np.linalg.norm((np.array([nodes[u][k] for k in 'zyx']) - np.array([nodes[v][k] for k in 'zyx'])) * S))})
    # forks
    res = DM.score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    tpf = {inv[x] for x in res.tp_forks}; fpf = {inv[x] for x in res.fp_forks}
    forks = []
    for p, cs in ch.items():
        if len(cs) < 2: continue
        st = []
        for c in cs:
            if p not in fidx or c not in fidx: st.append('synth')
            elif cpar.get(c) == p: st.append('agree')
            elif c in cpar: st.append('other_in_p13' if cpar[c] in nodes else 'other_dropped')
            else: st.append('nocand')
        forks.append({'lab': 'TP' if p in tpf else ('FP' if p in fpf else 'U'), 'st': sorted(st), 'fps': [cprob.get((p, c)) for c in cs],
                      'et': sorted(etype(next(e for e in edges if int(e['source_id']) == p and int(e['target_id']) == c)) for c in cs),
                      'ncand_ch': len(cch.get(p, []))})
    return {'set': s, 'movie': name, 'rows': rows, 'forks': forks, 'dev': [float(np.mean(dev)) if dev else -1, float(np.max(dev)) if dev else -1, len(dev), len(nodes)]}


if __name__ == '__main__':
    jobs = [(s, f) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(60) as p: R = p.map(job, jobs, chunksize=1)
    json.dump(R, open('/workspace/cl/ideas/up_rows.json', 'w'))
    print('done', len(R))
