"""Reviewer check (read-only): TP/FP/U labels of pp_longlink additions by length bin and embryo; GT edge-length stats."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import Counter
from multiprocessing import Pool
import numpy as np
import rule_eval as RE
S = np.array([1.625, .40625, .40625])

def labels(nodes, edges, gt):
    import evalx
    import tracksdata as td
    from tracking_cellmot import metrics as M
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    M._evaluate(pred, gt, 'jaccard', evalx.SCALE, 7.)
    ea = M._evaluate_matched_graph(pred, gt)
    K = td.DEFAULT_ATTR_KEYS
    lab = {}
    for s, d, m, v in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list(), ea[K.MATCHED_EDGE_MASK].to_list(), ea['pred_valid'].to_list()):
        lab[(inv[s], inv[d])] = 'TP' if m else ('FP' if v else 'U')
    return lab


def job(f):
    import evalx
    import tracksdata as td
    from ideas import pp_longlink
    name = Path(f).stem; st = [k for k in RE.FULL if ('_%s/' % k) in f][0]
    nodes, edges = evalx.load_graph_json(f)
    n2, e2, s2 = pp_longlink.apply(nodes, edges, name=name, set=st, fullgeff=RE.FULL[st] + '/' + name + '.geff', fe_min=0.5)
    gt, _ = evalx.load_gt(name)
    K = td.DEFAULT_ATTR_KEYS
    na = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    pos = {int(i): np.array([z, y, x]) for i, z, y, x in zip(na[K.NODE_ID].to_list(), na['z'].to_list(), na['y'].to_list(), na['x'].to_list())}
    ea = gt.edge_attrs()
    gl = [float(np.linalg.norm((pos[int(a)] - pos[int(b)]) * S)) for a, b in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list())]
    lab = labels(n2, e2, gt) if any('long_link' in e for e in e2) else {}
    out = []
    for e in e2:
        if 'long_link' not in e: continue
        a, b = int(e['source_id']), int(e['target_id'])
        d = float(np.linalg.norm((np.array([n2[a][k] for k in 'zyx'], float) - np.array([n2[b][k] for k in 'zyx'], float)) * S))
        out.append((round(d, 2), lab.get((a, b), 'miss'), e['long_link']))
    return dict(movie=name, set=st, add=out, gtlen=gl)

if __name__ == '__main__':
    fs = [f for s, d in RE.SRC['p13'].items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(48) as p: res = p.map(job, fs, chunksize=1)
    for emb in ['44b6', '6bba']:
        r = [x for x in res if x['movie'].startswith(emb)]
        g = np.array([d for x in r for d in x['gtlen']])
        print(emb, 'GT edges', len(g), '>14', int((g > 14).sum()), '>18', int((g > 18).sum()), '>22', int((g > 22).sum()), 'max', round(g.max(), 1), 'q99.9', round(np.quantile(g, .999), 1))
        for lo, hi in [(14, 16), (16, 18), (18, 20), (20, 25), (25, 1e9)]:
            c = Counter(l for x in r for d, l, p in x['add'] if lo < d <= hi)
            print('   add %g-%g' % (lo, hi), dict(c))
        c = Counter((l, x['set'] in ('hold36', 'prev4')) for x in r for d, l, p in x['add'])
        print('   by clean', dict(c))
    fp = sorted([(d, p, x['movie'], x['set']) for x in res for d, l, p in x['add'] if l == 'FP'])
    print('FP adds', fp)
    for L in ['TP','U','FP']:
        v=np.array([p for x in res for d,l,p in x['add'] if l==L]); print(L,'fe q', np.round(np.quantile(v,[0,.1,.25,.5,.75,.9]),3).tolist(), 'n', len(v))
    mv=Counter(x['movie'] for x in res for d,l,p in x['add'] if l=='TP'); print('TP movies', len(mv), sorted(mv.values(), reverse=True)[:8])
    tp = np.array([d for x in res for d, l, p in x['add'] if l == 'TP']); print('TP len max', tp.max(), 'n', len(tp))
