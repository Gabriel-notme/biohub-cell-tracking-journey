"""Dump evaluable candidate rows (features + GT label) of the union graph for all 199 P15 movies -> /workspace/cl/nm/gr_data/<set>__<movie>.npz"""
import sys; sys.path.insert(0, '/workspace/cl/nm')
import gr_common as C
import os, glob, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
OUT = Path('/workspace/cl/nm/gr_data'); OUT.mkdir(exist_ok=True)
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
RAD = float(os.environ.get('GR_RAD', '10'))


def job(a):
    s, f = a
    import numcodecs.blosc; numcodecs.blosc.use_threads = False
    import evalx, gr_feat
    name = Path(f).stem; out = OUT / ('%s__%s.npz' % (s, name))
    if out.exists(): return 0
    t0 = time.time()
    nodes, edges = evalx.load_graph_json(f)
    full = C.load_full(FULL[s] + '/' + name + '.geff')
    U = C.build_union(nodes, edges, full, radius=RAD, knn_all=True)
    L = C.gt_label(name, nodes, edges, U); C.label_edges(U, L)
    keys, X = gr_feat.features(U, nodes, edges, full, gr_feat.ilp_sets(s, name))
    ev = np.array([U['cand'][k]['ev'] for k in keys], bool); pos = np.array([U['cand'][k]['pos'] for k in keys], bool)
    np.savez_compressed(out, X=X[ev], pos=pos[ev], n_all=len(keys), n_p15=int(X[:, 0].sum()), sec=time.time() - t0,
                        a=np.array([k[0] for k in keys])[ev], b=np.array([k[1] for k in keys])[ev])
    return 1


if __name__ == '__main__':
    jobs = [(s, f) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s))]
    with Pool(24, maxtasksperchild=4) as p: print('done', sum(p.map(job, jobs, chunksize=1)), len(jobs))
