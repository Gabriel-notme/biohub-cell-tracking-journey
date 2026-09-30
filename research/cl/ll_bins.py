"""P14 risk check: precision of long_link and term_trim-join edges on annotated (evaluable) cells, by length and candidate prob,
per embryo and set. TP = GT edge between matched nodes; FP = evaluable (touches an annotated track) but not a GT edge."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']: os.environ.setdefault(_k, '1')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']


def job(f):
    import evalx
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    K = evalx.K
    name = Path(f).stem; s = f.split('/ps_p14_')[1].split('/')[0]
    nodes, edges = evalx.load_graph_json(f)
    sel = [e for e in edges if 'long_link' in e or 'dup_join' in e]
    if not sel: return []
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    ma = _matched_node_attrs(_match_full(pred, gt, evalx.SCALE, 7.))
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    ea = gt.edge_attrs(); GE = set(); gs = defaultdict(list); gp = defaultdict(list)
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): GE.add((int(x), int(y))); gs[int(x)].append(int(y)); gp[int(y)].append(int(x))
    S = np.array(evalx.SCALE); out = []
    for e in sel:
        a, b = int(e['source_id']), int(e['target_id'])
        ga, gb = p2g.get(a), p2g.get(b)
        ev = (ga is not None and bool(gs.get(ga))) or (gb is not None and bool(gp.get(gb)))
        lab = 'U' if not ev else ('TP' if (ga is not None and gb is not None and (ga, gb) in GE) else 'FP')
        d = float(np.linalg.norm((np.array([nodes[b][k] for k in 'zyx'], float) - np.array([nodes[a][k] for k in 'zyx'], float)) * S))
        out.append(dict(kind='long' if 'long_link' in e else 'join', emb=name[:4], set=s, d=d, p=float(e.get('long_link', 0)), lab=lab, movie=name))
    return out


if __name__ == '__main__':
    fs = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p14_%s/graphs/*.json' % s))]
    with Pool(48) as p: R = [r for rs in p.map(job, fs) for r in rs]
    json.dump(R, open('/workspace/cl/ll_bins.json', 'w'))
    for kind in ['long', 'join']:
        Q = [r for r in R if r['kind'] == kind]
        print('==', kind, 'total', len(Q), Counter(r['lab'] for r in Q))
        for emb in ['44b6', '6bba']:
            print('  ', emb, Counter(r['lab'] for r in Q if r['emb'] == emb))
        if kind == 'long':
            for lo, hi in [(14, 16), (16, 18), (18, 20), (20, 23), (23, 26), (26, 99)]:
                c = Counter(r['lab'] for r in Q if lo < r['d'] <= hi)
                print('   len (%d,%d]: n %5d TP %3d FP %2d U %5d  eval %.3f' % (lo, hi, sum(c.values()), c['TP'], c['FP'], c['U'], (c['TP'] + c['FP']) / max(1, sum(c.values()))))
            for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
                c = Counter(r['lab'] for r in Q if lo <= r['p'] < hi)
                print('   prob [%.1f,%.1f): n %5d TP %3d FP %2d U %5d' % (lo, hi, sum(c.values()), c['TP'], c['FP'], c['U']))
            fps = [r for r in Q if r['lab'] == 'FP']
            print('   FP details', [(r['movie'], round(r['d'], 1), r['p']) for r in fps])
