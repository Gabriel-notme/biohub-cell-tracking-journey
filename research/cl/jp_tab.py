"""Per-track table for junk pruning on a graph dir: jprune.features + GT labels (tp, fp edges in the track) + movie totals.
usage: jp_tab.py <tag> <graph_dir> -> /workspace/cl/jp/<tag>.pkl"""
import os, sys, json, glob, pickle
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from multiprocessing import Pool
import numpy as np


def job(f):
    name = Path(f).stem
    if not Path('/workspace/data/train/%s.geff/nodes/props' % name).exists(): return None
    import evalx, jprune
    from tracking_cellmot.metrics import evaluate, per_sample_metrics, node_recall
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    gt, n_total = evalx.load_gt(name)
    trk, X = jprune.features(nodes, edges)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    er = evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    row = per_sample_metrics(er, n_total, node_recall(pred, gt))
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    ea = gt.edge_attrs(); src = ea[K.EDGE_SOURCE].to_list(); dst = ea[K.EDGE_TARGET].to_list()
    ge = set(zip(src, dst)); gout = set(src); gin = set(dst)
    tp = np.zeros(len(trk)); fp = np.zeros(len(trk)); nm = np.zeros(len(trk))
    for i, c in enumerate(trk):
        nm[i] = sum(1 for n in c if n in p2g)
        for a, b in zip(c[:-1], c[1:]):
            if a in p2g and b in p2g and (p2g[a], p2g[b]) in ge: tp[i] += 1
            elif (a in p2g and p2g[a] in gout) or (b in p2g and p2g[b] in gin): fp[i] += 1
    return dict(movie=name, X=X, tp=tp, fp=fp, nm=nm, len=np.array([len(c) for c in trk]), TP=row['edge_tp'], FP=row['edge_fp'], FN=row['edge_fn'],
                ntot=float(n_total), npred=len(nodes))


if __name__ == '__main__':
    tag, gdir = sys.argv[1], sys.argv[2]
    with Pool(48) as p: R = [r for r in p.map(job, sorted(glob.glob(gdir + '/*.json'))) if r]
    Path('/workspace/cl/jp').mkdir(exist_ok=True)
    pickle.dump(R, open('/workspace/cl/jp/%s.pkl' % tag, 'wb'))
    print(tag, 'movies', len(R), 'tracks', sum(len(r['tp']) for r in R), 'track nodes', int(sum(r['len'].sum() for r in R)), 'of', sum(r['npred'] for r in R),
          'track TP', int(sum(r['tp'].sum() for r in R)), 'of', sum(r['TP'] for r in R), 'track FP', int(sum(r['fp'].sum() for r in R)), 'of', sum(r['FP'] for r in R))
