import sys, os, time, json
os.environ['BIOHUB_ART'] = '/workspace/art_b56/artifact_bundle'
sys.path.insert(0, '/workspace/art_b56/artifact_bundle'); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/code')
import numpy as np
import div_complete as dc
from refine_events import EventRefiner
from cell_event import chain, fork_geometry, edge_geometry
import evalx
name = sys.argv[1]; g = sys.argv[2]
nodes, edges = evalx.load_graph_json(g + '/' + name + '.json')
t0 = time.time()
rows, out, prev, pos = dc.candidates(nodes, edges); t1 = time.time()
print('candidates', len(rows), round(t1 - t0, 2), flush=True)
need = {x for r in rows for x in r[:4] if x is not None}
for (p, a, b, q, typ) in rows:
    for x in (p, a, b, q):
        if x is None: continue
        c = x
        for _ in range(6):
            c = prev.get(c)
            if c is None: break
            need.add(c)
t2 = time.time(); print('need', len(need), 'of', len(nodes), round(t2 - t1, 2), flush=True)
er = EventRefiner(['/workspace/art_b56/artifact_bundle/b1_best.pt', '/workspace/art_b56/artifact_bundle/b2_pretrained_best.pt'], '/workspace/data/train', {'fork_ensemble': 'first', 'edge_ensemble': 'last'}, '/workspace/cache/prof')
t3 = time.time(); print('load', round(t3 - t2, 2), flush=True)
sub = {n: nodes[n] for n in need}
ids, emb = er.embeddings(name, sub); lookup = {n: i for i, n in enumerate(ids)}
t4 = time.time(); print('emb', round(t4 - t3, 2), flush=True)
triples = [(p, a, b) for p, a, b, q, typ in rows]
cc = {}
def ch(n, d):
    k = (n, d is prev)
    if k not in cc: cc[k] = chain(n, d, pos)
    return cc[k]
fg = [fork_geometry(ch(p, prev), ch(a, out), ch(b, out)) for p, a, b in triples]
t5 = time.time(); print('geom', round(t5 - t4, 2), flush=True)
fp = er.score('fork', triples, fg, emb, lookup)
t6 = time.time(); print('fork score', round(t6 - t5, 2), flush=True)
