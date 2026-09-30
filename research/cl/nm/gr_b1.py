"""b1 edge-head probability for every evaluable union candidate (gr_data rows) of the clean40 movies (b1 never trained on them).
usage: gr_b1.py <gpu> <worker> <nworkers>  -> /workspace/cl/nm/gr_b1/<set>__<movie>.npy (aligned with gr_data rows)"""
import os, sys, time, glob, traceback
gpu, wi, nw = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']: os.environ[_k] = '2'
ART = '/workspace/art_b56/artifact_bundle'
sys.path.insert(0, '/workspace/cl/nm'); sys.path.insert(0, '/workspace/models/b34'); sys.path.insert(0, ART)
import numcodecs.blosc; numcodecs.blosc.use_threads = False
from pathlib import Path
from collections import defaultdict
import numpy as np
import gr_common as C
import evalx
from refine_events import EventRefiner
from cell_event import SCALE, chain, edge_geometry
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4'}
OUT = Path('/workspace/cl/nm/gr_b1'); OUT.mkdir(exist_ok=True)
jobs = [(s, f) for s in FULL for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s))][wi::nw]
er = EventRefiner([Path(ART) / 'b1_best.pt'], '/workspace/data/train', {'edge_ensemble': 'last'}, Path('/workspace/cl/nm/gr_b1_cache_%d' % wi))
for s, f in jobs:
    name = Path(f).stem; dst = OUT / ('%s__%s.npy' % (s, name))
    if dst.exists(): continue
    t0 = time.time()
    try:
        nodes, edges = evalx.load_graph_json(f)
        full = C.load_full(FULL[s] + '/' + name + '.geff')
        U = C.build_union(nodes, edges, full, radius=10, knn_all=True)
        z = np.load('/workspace/cl/nm/gr_data/%s__%s.npz' % (s, name))
        A, B = z['a'], z['b']
        ids = U['ids']
        allnodes = {int(ids[i]): {'t': int(U['t'][i]), 'z': float(U['xyz'][i][0]), 'y': float(U['xyz'][i][1]), 'x': float(U['xyz'][i][2])} for i in set(A.tolist()) | set(B.tolist())}
        out = defaultdict(list); prev = {}
        for e in edges:
            a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); prev[b] = a
        pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
        for n, v in allnodes.items(): pos.setdefault(n, np.array([v['z'], v['y'], v['x']], np.float32) * SCALE)
        pairs = [(int(ids[a]), int(ids[b])) for a, b in zip(A.tolist(), B.tolist())]
        eids, emb = er.embeddings(name, allnodes); lookup = {n: i for i, n in enumerate(eids)}
        geo = [edge_geometry(chain(a, prev, pos), chain(b, out, pos)) for a, b in pairs]
        pr = er.score('edge', pairs, geo, emb, lookup)
        np.save(dst, pr.astype(np.float32))
        print('OK', s, name, len(pairs), len(allnodes), round(time.time() - t0, 1), flush=True)
    except Exception as e:
        print('ERR', s, name, repr(e)[:200], traceback.format_exc()[-800:], flush=True)
print('WORKER_DONE', wi, flush=True)
