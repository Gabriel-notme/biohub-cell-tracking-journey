import os, sys, json, glob
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
import numpy as np

def job(path):
    import evalx
    from scipy.spatial import cKDTree
    name = Path(path).stem
    nodes, edges = evalx.load_graph_json(path)
    row = evalx.score_movie(name, nodes, edges, detail=True)
    # unmatched GT node distance to nearest prediction (same t)
    gt, _ = evalx.load_gt(name)
    ga = gt.node_attrs(attr_keys=['t', 'z', 'y', 'x'])
    S = np.array(evalx.SCALE)
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append([round(v['z']), round(v['y']), round(v['x'])])
    trees = {t: cKDTree(np.array(p, float) * S) for t, p in byt.items()}
    d = []
    for t, z, y, x in zip(ga['t'].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list()):
        tr = trees.get(int(t))
        if tr is None: d.append(99.); continue
        dd, _ = tr.query(np.array([z, y, x], float) * S); d.append(float(dd))
    d = np.array(d)
    row['gt_nn_hist'] = [int((d < 3).sum()), int(((d >= 3) & (d < 5)).sum()), int(((d >= 5) & (d < 7)).sum()), int(((d >= 7) & (d < 10)).sum()), int((d >= 10).sum())]
    row['n_gt_nodes'] = len(d)
    return row

paths = sorted(glob.glob(sys.argv[1] + '/*.json'))
with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as pool: rows = pool.map(job, paths, chunksize=1)
cat = Counter(); hist = np.zeros(5, int)
for r in rows:
    cat.update(r['analysis']); hist += np.array(r['gt_nn_hist'])
print('movies', len(rows))
print('categories', dict(sorted(cat.items())))
print('GT nearest-pred distance hist [<3,3-5,5-7,7-10,>=10]:', hist.tolist())
tot = sum(r['edge_tp'] + r['edge_fp'] + r['edge_fn'] for r in rows)
for r in sorted(rows, key=lambda r: -(r['edge_fp'] + r['edge_fn'])):
    a = r['analysis']
    print('%s tp=%d fp=%d fn=%d ratio=%.3f npred=%d recall=%.3f nn=%s swap=%d gapfree=%d srcend=%d dststart=%d miss=%d' % (r['movie'], r['edge_tp'], r['edge_fp'], r['edge_fn'], r['total_node_ratio'], r['num_pred_nodes'], r['node_recall'], r['gt_nn_hist'], a.get('fn_swap', 0), a.get('fn_gap_both_free', 0), a.get('fn_src_ends_dst_taken', 0), a.get('fn_src_elsewhere_dst_starts', 0), a.get('fn_src_missing', 0) + a.get('fn_dst_missing', 0) + a.get('fn_both_missing', 0)))
json.dump(rows, open(sys.argv[3] if len(sys.argv) > 3 else '/workspace/runs/agg.json', 'w'))
