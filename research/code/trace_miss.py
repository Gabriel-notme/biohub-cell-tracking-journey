import os, sys, json, glob
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
os.environ['POLARS_MAX_THREADS'] = '2'
import numpy as np, zarr
from scipy.spatial import cKDTree
S = np.array([1.625, .40625, .40625])
RUN = '/workspace/runs/b5_hold36'
FINAL = '/workspace/runs/pstage_test_hold36/graphs'
def load_json_nodes(p):
    d = json.load(open(p)); return {int(k): (int(v['t']), v['z'], v['y'], v['x']) for k, v in d['nodes'].items()}, d['edges']
def load_geff_nodes(p):
    g = zarr.open_group(p, mode='r'); P = {k: np.asarray(g['nodes/props/%s/values' % k][:]) for k in 'tzyx'}
    ids = np.asarray(g['nodes/ids'][:])
    return {int(i): (int(P['t'][j]), float(P['z'][j]), float(P['y'][j]), float(P['x'][j])) for j, i in enumerate(ids)}, None
def job(name):
    import evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    nodes, edges = evalx.load_graph_json(FINAL + '/' + name + '.json')
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    mg = {int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    um = [(int(t), np.array([z, y, x]) * S) for i, t, z, y, x in zip(*[ga[c].to_list() for c in [K.NODE_ID, 't', 'z', 'y', 'x']]) if int(i) not in mg]
    stages = [('raw', RUN + '/working/tracking_repo/predictions/unknown/unet_transformer/split_0/%s.geff' % name), ('ref', RUN + '/working/reference_graphs/%s.json' % name)]
    for sd in sorted(glob.glob(RUN + '/stages/*')): stages.append((Path(sd).name, sd + '/%s.json' % name))
    stages.append(('p1', FINAL + '/%s.json' % name))
    res = {}
    for sname, p in stages:
        if not os.path.exists(p): continue
        N, _ = load_geff_nodes(p) if p.endswith('.geff') else load_json_nodes(p)
        byt = defaultdict(list)
        for k, v in N.items(): byt[v[0]].append(v[1:])
        trees = {t: cKDTree(np.array(v) * S) for t, v in byt.items()}
        ds = [float(trees[t].query(q)[0]) if t in trees else 99. for t, q in um]
        res[sname] = [int(sum(d <= 7 for d in ds)), int(sum(d <= 5 for d in ds)), len(N)]
    return name, len(um), res
if __name__ == '__main__':
    names = [l.strip() for l in open('/workspace/hold36.txt') if l.strip()]
    with Pool(36) as pool: out = pool.map(job, names)
    tot = Counter(); tot5 = Counter(); nn = Counter(); U = 0
    for name, u, res in out:
        U += u
        for s, (a, b, n) in res.items(): tot[s] += a; tot5[s] += b; nn[s] += n
    print('unmatched GT nodes (final P1):', U)
    for s in tot: print('%-26s node<=7um %4d  <=5um %4d   total nodes %d' % (s, tot[s], tot5[s], nn[s]))
