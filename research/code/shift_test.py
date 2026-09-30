import os, sys, json
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
os.environ['POLARS_MAX_THREADS'] = '2'
import numpy as np
S = np.array([1.625, .40625, .40625])
def job(a):
    path, sh = a
    import evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    name = Path(path).stem
    nodes, edges = evalx.load_graph_json(path)
    nn = {k: dict(v, z=v['z'] + sh[0], y=v['y'] + sh[1], x=v['x'] + sh[2]) for k, v in nodes.items()}
    row = evalx.score_movie(name, nn, edges)
    row['shift'] = list(sh)
    if sh == (0., 0., 0.):
        gt, _ = evalx.load_gt(name)
        pred, mapping = evalx.to_graph(nodes, edges, rounding=False)
        evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
        inv = {v: k for k, v in mapping.items()}
        na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
        ga = gt.node_attrs(attr_keys=[K.NODE_ID, 'z', 'y', 'x'])
        gpos = {int(i): np.array([z, y, x]) for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
        D = [np.array([nodes[inv[int(a)]][c] for c in 'zyx']) - gpos[int(b)] for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1]
        D = np.array(D); row['disp_mean_vox'] = D.mean(0).tolist(); row['disp_n'] = len(D)
    return row
if __name__ == '__main__':
    gdir = sys.argv[1]; outp = sys.argv[2]
    shifts = [tuple(float(x) for x in s.split(',')) for s in sys.argv[3].split(';')]
    files = sorted(Path(gdir).glob('*.json'))
    jobs = [(str(f), s) for s in shifts for f in files]
    with Pool(min(120, len(jobs))) as pool: rows = pool.map(job, jobs, chunksize=1)
    json.dump(rows, open(outp, 'w'))
    from tracking_cellmot.metrics import summarise
    for s in shifts:
        rr = [r for r in rows if tuple(r['shift']) == s]
        sm = summarise(rr)
        print(s, 'score %.6f adjE %.6f E %.6f div %.4f tp %d fp %d fn %d' % (sm['score'], sm['adj_edge_jaccard'], sm['edge_jaccard'], sm['division_jaccard'], sum(r['edge_tp'] for r in rr), sum(r['edge_fp'] for r in rr), sum(r['edge_fn'] for r in rr)))
