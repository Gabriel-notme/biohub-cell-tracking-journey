"""b1 node embeddings for all final P5-style nodes and all dropped pre-ILP detections, with GT labels.
usage: emb_extract.py <gpu> <worker> <nworkers>"""
import os, sys, json, time, traceback
gpu, wi, nw = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu); os.environ['OMP_NUM_THREADS'] = '4'; os.environ['POLARS_MAX_THREADS'] = '2'
ART = '/workspace/art_b56/artifact_bundle'; os.environ['BIOHUB_ART'] = ART
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, ART); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
SETS = {'t127a': ('/workspace/t127a.txt', '/workspace/cl/p5tr/t127a', '/workspace/sync4/runs/fullgraph_t127a'),
        't127b': ('/workspace/t127b.txt', '/workspace/cl/p5tr/t127b', '/workspace/sync3/runs/fullgraph_t127b'),
        'audit32': ('/workspace/audit32.txt', '/workspace/cl/p5tr/audit32', '/workspace/sync3/runs/fullgraph_audit32'),
        'hold36': ('/workspace/hold36.txt', '/workspace/cl/ps_p5_hold36/graphs', '/workspace/runs/fullgraph_hold36'),
        'prev4': ('/workspace/preview4.txt', '/workspace/cl/ps_p5_prev4/graphs', '/workspace/runs/fullgraph_prev4')}
OUT = Path('/workspace/cl/emb'); OUT.mkdir(exist_ok=True)
S = np.array([1.625, .40625, .40625]); OFF = 10 ** 12
jobs = [(s, n, g, f) for s, (lst, g, f) in SETS.items() for n in [l.strip() for l in open(lst) if l.strip()]]
jobs = sorted(jobs, key=lambda j: -(Path(j[2]) / (j[1] + '.json')).stat().st_size)[wi::nw]
cfg = json.load(open('/workspace/p12ds/p3_config.json'))
from refine_events import EventRefiner
import evalx, edge_link
K = evalx.K
er = EventRefiner([Path(ART) / n for n in cfg['models']], '/workspace/data/train', cfg.get('event_config'), Path('/workspace/cl/ev_cache_emb_%d' % wi))
for s, name, g, f in jobs:
    dst = OUT / ('%s__%s.npz' % (s, name))
    if dst.exists(): continue
    t0 = time.time()
    try:
        nodes, edges = evalx.load_graph_json(Path(g) / (name + '.json'))
        fids, fT, fV, fE, fprob = edge_link.load_full(Path(f) / (name + '.geff'))
        drop = [j for j, i in enumerate(fids.tolist()) if int(i) not in nodes]
        allnodes = {n: {'t': int(v['t']), 'z': float(v['z']), 'y': float(v['y']), 'x': float(v['x'])} for n, v in nodes.items()}
        for j in drop:
            allnodes[OFF + int(fids[j])] = {'t': int(fT[j]), 'z': float(fV[j][0]), 'y': float(fV[j][1]), 'x': float(fV[j][2])}
        ids, emb = er.embeddings(name, allnodes)
        E = emb[0].astype(np.float16)
        # GT labels
        gt, n_total = evalx.load_gt(name)
        pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
        from tracking_cellmot.metrics import evaluate
        evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
        na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
        p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
        matched_g = set(p2g.values())
        ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
        gall = defaultdict(list); gun = defaultdict(list)
        for i, t, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['t'].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list()):
            p = np.array([z, y, x]) * S; gall[int(t)].append(p)
            if int(i) not in matched_g: gun[int(t)].append(p)
        ta = {t: cKDTree(np.array(v)) for t, v in gall.items()}; tu = {t: cKDTree(np.array(v)) for t, v in gun.items()}
        kind = np.array([1 if n >= OFF else 0 for n in ids], np.int8)
        T = np.array([allnodes[n]['t'] for n in ids], np.int32)
        P = np.array([[allnodes[n][k] for k in 'zyx'] for n in ids], np.float32)
        matched = np.array([1 if (n < OFF and n in p2g) else 0 for n in ids], np.int8)
        d_any = np.full(len(ids), 99., np.float32); d_un = np.full(len(ids), 99., np.float32)
        for t in np.unique(T):
            m = T == t
            if int(t) in ta: d_any[m] = ta[int(t)].query(P[m] * S)[0]
            if int(t) in tu: d_un[m] = tu[int(t)].query(P[m] * S)[0]
        np.savez_compressed(dst, ids=np.array(ids, np.int64), kind=kind, emb=E, t=T, pos=P, matched=matched, d_any=d_any, d_un=d_un, n_total=np.float64(n_total))
        for p in Path('/workspace/cl/ev_cache_emb_%d' % wi).glob('*'): p.unlink()
        print('OK', s, name, len(ids), int(kind.sum()), E.shape, round(time.time() - t0, 1), flush=True)
    except Exception as e:
        print('ERR', s, name, repr(e)[:200], traceback.format_exc()[-800:], flush=True)
print('WORKER_DONE', wi, flush=True)
