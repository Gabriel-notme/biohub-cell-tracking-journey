"""Where are annotated cells vs predicted nodes? z (relative to volume depth), time, and N_total vs N_pred per embryo.
If annotation systematically avoids a region in BOTH embryos, pred nodes there cost only the node-count factor."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from multiprocessing import Pool
import numpy as np
SETS = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
        'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
        't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}


def job(f):
    import evalx, zarr
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    gt, ntot = evalx.load_gt(name)
    sh = zarr.open_group('/workspace/data/train/%s.zarr' % name, mode='r')['0'].shape  # T,Z,Y,X
    na = gt.node_attrs(attr_keys=['t', 'z', 'y', 'x'])
    gz = np.array(na['z'].to_list()) / sh[1]; gt_ = np.array(na['t'].to_list()) / max(1, sh[0] - 1)
    gy = np.array(na['y'].to_list()) / sh[2]; gx = np.array(na['x'].to_list()) / sh[3]
    pz = np.array([v['z'] for v in nodes.values()]) / sh[1]; pt = np.array([v['t'] for v in nodes.values()]) / max(1, sh[0] - 1)
    return name, sh, ntot, len(nodes), np.histogram(gz, bins=10, range=(0, 1))[0], np.histogram(pz, bins=10, range=(0, 1))[0], \
        np.histogram(gt_, bins=10, range=(0, 1))[0], np.histogram(pt, bins=10, range=(0, 1))[0], float(gy.min()), float(gy.max()), float(gx.min()), float(gx.max())


if __name__ == '__main__':
    fs = [f for s in SETS for f in sorted(glob.glob(SETS[s] + '/*.json'))]
    with Pool(64) as p: R = p.map(job, fs)
    print('shapes', sorted(set(tuple(r[1]) for r in R))[:10])
    for emb in ['44b6', '6bba']:
        Q = [r for r in R if r[0].startswith(emb)]
        gz = sum(r[4] for r in Q); pz = sum(r[5] for r in Q); gt_ = sum(r[6] for r in Q); pt = sum(r[7] for r in Q)
        print('==', emb, 'movies', len(Q), 'sum Npred %d sum Ntotal %d' % (sum(r[3] for r in Q), sum(r[2] for r in Q)))
        print('  z decile  GT %s' % np.round(gz / gz.sum(), 3).tolist()); print('            pred %s' % np.round(pz / pz.sum(), 3).tolist())
        print('  t decile  GT %s' % np.round(gt_ / gt_.sum(), 3).tolist()); print('            pred %s' % np.round(pt / pt.sum(), 3).tolist())
        ratio = np.array([(r[3] - r[2]) / r[2] for r in Q])
        print('  (Npred-Ntot)/Ntot p10 %.3f p50 %.3f p90 %.3f' % tuple(np.percentile(ratio, [10, 50, 90])))
