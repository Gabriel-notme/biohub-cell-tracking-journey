"""Synthetic edge cases for jprune.apply + the p_stage6 check_graph fallback path, and timing on the largest real graph."""
import sys, json, time, traceback
import numpy as np
sys.path.insert(0, '/workspace/p56stage')
import jprune
import importlib.util
spec = importlib.util.spec_from_file_location('ps6', '/workspace/p56stage/p_stage6.py'); ps6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps6)
M = '/workspace/p56stage/jp_lgb.json'
cfg = json.load(open('/workspace/p56stage/p9_config.json'))['junk_prune']


def run(tag, nodes, edges):
    try:
        n, e, st = jprune.apply(nodes, edges, M, lam=cfg['lam'], tp_ref=cfg['tp_ref'], ratio=cfg['ratio'])
        try:
            ps6.check_graph(n, e, nodes); chk = 'check_graph OK'
        except AssertionError as ex:
            chk = 'check_graph FAIL(%s) -> fallback to pre-junk graph' % ex
        print('%-40s nodes %d->%d edges %d->%d %s %s' % (tag, len(nodes), len(n), len(edges), len(e), st, chk))
    except Exception as ex:
        print('%-40s EXCEPTION %r (caught by p_stage6 try -> fallback)' % (tag, ex))


def node(i, t, z, y, x): return {'node_id': i, 't': t, 'z': z, 'y': y, 'x': x}

run('empty graph', {}, [])
run('single node', {1: node(1, 0, 5, 50, 50)}, [])
run('one node per frame, no edges (T=5)', {i: node(i, i, 5, 50, 50) for i in range(5)}, [])
# a single long track spanning all frames far from everything
nodes = {i: node(i, i, 5, 10 + 0.1 * i, 10) for i in range(10)}
edges = [{'source_id': i, 'target_id': i + 1, 'edge_prob': 0.5} for i in range(9)]
run('single track all frames', nodes, edges)
# many isolated 2-node tracks, one per frame pair, sparse movie
rng = np.random.default_rng(0)
nodes = {}; edges = []; k = 0
for t in range(0, 20, 2):
    for j in range(3):
        a, b = k, k + 1; k += 2
        p = rng.uniform(0, 250, 2)
        nodes[a] = node(a, t, 3, *p); nodes[b] = node(b, t + 1, 3, *(p + 0.5))
        edges.append({'source_id': a, 'target_id': b, 'edge_prob': 0.1})
run('sparse 2-node junk tracks (all isolated)', nodes, edges)
# NaN coordinate
nodes2 = dict(nodes); nodes2[0] = node(0, 0, float('nan'), 10, 10)
run('NaN coordinate', nodes2, edges)
# edge_prob None / missing
edges3 = [dict(e, edge_prob=None) for e in edges]
run('edge_prob None', nodes, edges3)
edges4 = [{'source_id': e['source_id'], 'target_id': e['target_id']} for e in edges]
run('edge_prob key missing', nodes, edges4)
# string ids / float t
nodes5 = {a: dict(v, t=float(v['t'])) for a, v in nodes.items()}
run('float t', nodes5, edges)
# 2D-ish: z all equal (zptp=0)
nodes6 = {a: dict(v, z=0.0) for a, v in nodes.items()}
run('z constant', nodes6, edges)
# timing on largest real P8 graph
import glob, os
fs = sorted(glob.glob('/workspace/cl/ps_p8_*/graphs/*.json'), key=os.path.getsize)[-1]
g = json.load(open(fs)); N = {int(k): v for k, v in g['nodes'].items()}
t0 = time.time(); trk, X = jprune.features(N, g['edges']); t1 = time.time()
n, e, st = jprune.apply(N, g['edges'], M, lam=cfg['lam'], tp_ref=cfg['tp_ref'], ratio=cfg['ratio']); t2 = time.time()
print('largest graph', fs, 'nodes', len(N), 'tracks', len(trk), 'features %.1fs, full apply %.1fs' % (t1 - t0, t2 - t1), st)
print('nan in X:', int(np.isnan(X).sum()), 'inf in X:', int(np.isinf(X).sum()))
t0 = time.time(); import lgb_np; m = lgb_np.load(M); print('model load %.2fs' % (time.time() - t0))
