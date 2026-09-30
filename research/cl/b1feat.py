"""b1 edge-head probabilities for edge_link (gap1) and relink candidate pairs on P3 graphs.
usage: b1feat.py <gpu> <worker> <nworkers>"""
import os, sys, json, time, traceback
gpu, wi, nw = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu); os.environ['OMP_NUM_THREADS'] = '4'
ART = '/workspace/art_b56/artifact_bundle'; os.environ['BIOHUB_ART'] = ART
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, ART); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict
import numpy as np
SETS = {'hold36': ('/workspace/hold36.txt', '/workspace/runs/p3_hold36/graphs'), 'prev4': ('/workspace/preview4.txt', '/workspace/runs/p3_prev4/graphs'),
        'audit32': ('/workspace/audit32.txt', '/workspace/sync3/runs/p3_audit32/graphs'), 't127a': ('/workspace/t127a.txt', '/workspace/sync4/runs/fin_p3_t127a/graphs'),
        't127b': ('/workspace/t127b.txt', '/workspace/sync3/runs/fin_p3_t127b/graphs')}
OUT = Path('/workspace/cl/b1e'); OUT.mkdir(exist_ok=True)
jobs = [(s, n, g) for s, (lst, g) in SETS.items() for n in [l.strip() for l in open(lst) if l.strip()]]
jobs = sorted(jobs, key=lambda j: -(Path(j[2]) / (j[1] + '.json')).stat().st_size)[wi::nw]
cfg = json.load(open('/workspace/p12ds/p3_config.json'))
from refine_events import EventRefiner
from cell_event import SCALE, chain, edge_geometry
er = EventRefiner([Path(ART) / n for n in cfg['models']], '/workspace/data/train', cfg.get('event_config'), Path('/workspace/cl/ev_cache_b1e_%d' % wi))
for s, name, g in jobs:
    dst = OUT / ('%s__%s.json' % (s, name))
    if dst.exists(): continue
    t0 = time.time()
    try:
        d = json.load(open(Path(g) / (name + '.json')))
        nodes = {int(k): v for k, v in d['nodes'].items()}
        out = defaultdict(list); prev = {}
        for e in d['edges']:
            a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); prev[b] = a
        pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
        pairs = set()
        for r in json.load(open('/workspace/cl/ecands/%s__%s.json' % (s, name))):
            if r['gap'] == 1: pairs.add((r['s'], r['d']))
        for r in json.load(open('/workspace/cl/rcands/%s__%s.json' % (s, name))):
            pairs.add((r['s'], r['d']))
            if r['cur_d'] is not None: pairs.add((r['s'], r['cur_d']))
            if r['cur_s'] is not None: pairs.add((r['cur_s'], r['d']))
        pairs = sorted(p for p in pairs if p[0] in nodes and p[1] in nodes)
        res = {}
        if pairs:
            sub = {n: nodes[n] for p in pairs for n in p}
            ids, emb = er.embeddings(name, sub); lookup = {n: i for i, n in enumerate(ids)}
            geo = [edge_geometry(chain(a, prev, pos), chain(b, out, pos)) for a, b in pairs]
            pr = er.score('edge', pairs, geo, emb, lookup)
            res = {'%d_%d' % p: float(x) for p, x in zip(pairs, pr)}
        dst.write_text(json.dumps(res))
        print('OK', s, name, len(pairs), round(time.time() - t0, 1), flush=True)
    except Exception as e:
        print('ERR', s, name, repr(e)[:200], traceback.format_exc()[-600:], flush=True)
print('WORKER_DONE', wi, flush=True)
