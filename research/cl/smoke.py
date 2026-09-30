import sys, json, time
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/code')
import evalx
from ideas import p19_edge as m
nodes, edges = evalx.load_graph_json('/workspace/cl/p16/ps_p17_hold36/graphs/44b6_144b256d.json')
for kw in [dict(tstub=1), dict(border=1), dict(border=1, border_mode='any'), dict(padd=1)]:
    t = time.time()
    nn, ne, st = m.apply(nodes, edges, name='44b6_144b256d', zarr='/workspace/data/train/44b6_144b256d.zarr', set='hold36', fullgeff='x', **kw)
    print(kw, st, len(nn), len(ne), round(time.time() - t, 2))
