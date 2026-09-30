"""Re-solve the base ILP from saved pre-ILP candidate graphs with different weights.
usage: resolve_ilp.py <full_dir> <out_dir> <repo_dir> <appearance> <disappearance> <division> [edge_weight] [nproc]
"""
import os, sys, glob, time
from pathlib import Path
from multiprocessing import Pool
full_dir, out_dir, repo = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
A, D, V = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
EW = float(sys.argv[7]) if len(sys.argv) > 7 else -1.0
NP = int(sys.argv[8]) if len(sys.argv) > 8 else 16
def job(p):
    sys.path.insert(0, repo + '/src'); sys.path.insert(0, repo + '/scripts')
    os.environ['POLARS_MAX_THREADS'] = '2'
    import numpy as np, polars as pl, zarr, tracksdata as td
    from biohub_tracking.io import save_graph
    t0 = time.time()
    g = zarr.open_group(p, mode='r')
    ids = np.asarray(g['nodes/ids'][:]); P = {k: np.asarray(g['nodes/props/%s/values' % k][:]) for k in 'tzyx'}
    E = np.asarray(g['edges/ids'][:]); ep = np.asarray(g['edges/props/edge_prob/values'][:]); ed = np.asarray(g['edges/props/edge_dist/values'][:])
    order = np.argsort(ids)
    graph = td.graph.InMemoryGraph()
    for k in ['z', 'y', 'x']: graph.add_node_attr_key(k, pl.Float64, -999999.0)
    nids = graph.bulk_add_nodes([{'t': int(P['t'][j]), 'z': float(P['z'][j]), 'y': float(P['y'][j]), 'x': float(P['x'][j])} for j in order])
    m = {int(ids[j]): n for j, n in zip(order, nids)}
    graph.add_edge_attr_key('edge_prob', pl.Float64, 0.0); graph.add_edge_attr_key('edge_dist', pl.Float64, 0.0)
    graph.bulk_add_edges([{'source_id': m[int(s)], 'target_id': m[int(t)], 'edge_prob': float(pp), 'edge_dist': float(dd)} for (s, t), pp, dd in zip(E.tolist(), ep.tolist(), ed.tolist())])
    solver = td.solvers.ILPSolver(edge_weight=EW * td.EdgeAttr('edge_prob'), appearance_weight=A, disappearance_weight=D, division_weight=V)
    sol = solver.solve(graph)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_graph(sol, out_dir / Path(p).name)
    return Path(p).name, len(ids), sol.num_nodes(), sol.num_edges(), round(time.time() - t0, 1)
if __name__ == '__main__':
    ps = sorted(glob.glob(full_dir + '/*.geff'))
    with Pool(NP) as pool:
        for r in pool.imap_unordered(job, ps): print('ILP', *r, flush=True)
    print('ILP_DONE')
