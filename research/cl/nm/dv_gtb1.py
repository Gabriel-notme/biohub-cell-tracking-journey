"""dv_gtb1 (diagnostic, reads GT): generalisation gap of the b1 fork head, GT-anchored and independent of the tracker.
Positives: every GT division whose parent d (t) and both children (t+1) are matched to P15 nodes -> triple (P, A, B).
Negatives: annotated non-dividing GT cells g (one child) -> (P=pred(g), A=pred(child), B = each other P15 node at t+1 within 13 um of P);
up to 150 per movie. Scores b1 fork prob; reports AUC for b1-unseen movies (hold36+prev4) vs b1-training movies (audit32+t127).
phase 'patch' (CPU) then 'gpu' <dev>."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
ART = '/workspace/art_b56/artifact_bundle'; os.environ['BIOHUB_ART'] = ART
for p in ['/workspace/cl', '/workspace/official/src', '/workspace/code', ART]:
    sys.path.insert(0, p)
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
PD = Path('/workspace/cl/nm/dv_gtpatch')
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']


def job(a):
    s, name = a
    of = PD / ('%s__%s.npz' % (s, name))
    if of.exists(): return 1
    import numcodecs.blosc
    numcodecs.blosc.use_threads = False
    import evalx
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    from cell_event import SCALE, Movie, chain, fork_geometry
    from scipy.spatial import cKDTree
    K = evalx.K
    rng = np.random.default_rng(abs(hash(name)) % 2 ** 31)
    nodes, edges = evalx.load_graph_json('/workspace/cl/ps_p15_%s/graphs/%s.json' % (s, name))
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(x)]: int(y) for x, y in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    g2p = {y: x for x, y in p2g.items()}
    ea = gt.edge_attrs(); gs = defaultdict(list)
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(x)].append(int(y))
    out, prev = defaultdict(list), {}
    for e in edges:
        x, y = int(e['source_id']), int(e['target_id']); out[x].append(y); prev[y] = x
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
    tt = {n: int(v['t']) for n, v in nodes.items()}
    frames = defaultdict(list)
    for n in nodes: frames[tt[n]].append(n)
    trees = {t: cKDTree(np.array([pos[n] for n in ns])) for t, ns in frames.items()}
    trip, lab = [], []
    for g, ch in gs.items():
        if len(ch) >= 2 and g in g2p and ch[0] in g2p and ch[1] in g2p:
            trip.append((g2p[g], g2p[ch[0]], g2p[ch[1]])); lab.append(1)
    neg = []
    for g, ch in gs.items():
        if len(ch) != 1 or g not in g2p or ch[0] not in g2p: continue
        P, A = g2p[g], g2p[ch[0]]
        if tt[A] != tt[P] + 1: continue
        for j in trees[tt[P] + 1].query_ball_point(pos[P], 13.0):
            B = frames[tt[P] + 1][j]
            if B != A: neg.append((P, A, B))
    if len(neg) > 150: neg = [neg[i] for i in rng.choice(len(neg), 150, replace=False)]
    trip += neg; lab += [0] * len(neg)
    if not trip: np.savez(of, n=0); return 1
    need = sorted({x for tr in trip for x in tr}, key=lambda n: (tt[n], n)); idx = {n: i for i, n in enumerate(need)}
    mv = Movie(Path('/workspace/data/train') / (name + '.zarr'), context=5)
    X = mv.patches([tt[n] for n in need], [[nodes[n][k] for k in 'zyx'] for n in need])
    # geometry as if (P;A,B) were a fork: P history, A and B forward chains in the current graph
    G = [fork_geometry(chain(P, prev, pos), chain(A, out, pos), chain(B, out, pos)) for P, A, B in trip]
    np.savez(of, n=len(trip), X=X, G=np.stack(G).astype(np.float32), T=np.array([[idx[x] for x in tr] for tr in trip]), y=np.array(lab),
             d_pb=np.array([float(np.linalg.norm(pos[P] - pos[B])) for P, A, B in trip]))
    return 1


def gpu(dev):
    os.environ['CUDA_VISIBLE_DEVICES'] = dev
    import torch
    from cell_event import load_event_model
    from sklearn.metrics import roc_auc_score
    model, _ = load_event_model(Path(ART) / 'b1_best.pt', 'cuda')
    res = defaultdict(lambda: ([], []))
    for f in sorted(PD.glob('*.npz')):
        z = np.load(f)
        if int(z['n']) == 0: continue
        s, name = f.stem.split('__')
        X = torch.from_numpy(z['X']).cuda().float()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            E = torch.cat([model.encode(X[i:i + 512]).float() for i in range(0, len(X), 512)])
            T = torch.from_numpy(z['T']).cuda()
            lg = model.fork_logits(E[T[:, 0]], E[T[:, 1]], E[T[:, 2]], torch.from_numpy(z['G']).cuda()).float().cpu().numpy()
        g = 'unseen' if s in ('hold36', 'prev4') else 'b1train'
        for key in [(name[:4], g), ('all', g)]:
            res[key][0].extend(lg.tolist()); res[key][1].extend(z['y'].tolist())
    for k in sorted(res):
        sc, y = np.array(res[k][0]), np.array(res[k][1])
        pr = 1 / (1 + np.exp(-sc))
        print(k, 'pos %d neg %d AUC %.3f | pos with p>=0.9: %.2f  neg with p>=0.9: %.4f | pos median p %.3f' % (
            y.sum(), (1 - y).sum(), roc_auc_score(y, sc), (pr[y == 1] >= .9).mean(), (pr[y == 0] >= .9).mean(), np.median(pr[y == 1])))


if __name__ == '__main__':
    if sys.argv[1] == 'patch':
        PD.mkdir(parents=True, exist_ok=True)
        jobs = [(s, Path(f).stem) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s))]
        with Pool(24, maxtasksperchild=2) as p: print('patch done', sum(p.map(job, jobs, chunksize=1)))
    else:
        gpu(sys.argv[2])
