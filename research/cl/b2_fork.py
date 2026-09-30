"""b2 (pretrained temporal U-Net event model) fork probabilities for division-completion candidates on B5 base graphs.
usage: b2_fork.py <gpu> <worker> <nworkers>"""
import os, sys, json, time, traceback
gpu, wi, nw = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu); os.environ['OMP_NUM_THREADS'] = '4'
ART = '/workspace/art_b56/artifact_bundle'; os.environ['BIOHUB_ART'] = ART
os.environ['BIOHUB_BASE_REPO'] = '/workspace/runs/b5f_hold36/working/tracking_repo'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, ART); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict
import numpy as np
SETS = {'hold36': ('/workspace/hold36.txt', '/workspace/runs/b5f_hold36/working/lineage_graphs'),
        'prev4': ('/workspace/preview4.txt', '/workspace/runs/b5f_prev4/working/lineage_graphs'),
        'audit32': ('/workspace/audit32.txt', '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs'),
        't127a': ('/workspace/t127a.txt', '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs'),
        't127b': ('/workspace/t127b.txt', '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs')}
OUT = Path('/workspace/cl/b2f'); OUT.mkdir(exist_ok=True)
jobs = [(s, n, g) for s, (lst, g) in SETS.items() for n in [l.strip() for l in open(lst) if l.strip()]]
jobs = sorted(jobs, key=lambda j: -(Path(j[2]) / (j[1] + '.json')).stat().st_size)[wi::nw]
from refine_events import EventRefiner
from cell_event import SCALE, chain, fork_geometry
er = EventRefiner([Path(ART) / 'b2_pretrained_best.pt'], '/workspace/data/train', {'fork_ensemble': 'first', 'edge_ensemble': 'last'}, Path('/workspace/cl/ev_cache_b2_%d' % wi))
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
        rows = json.load(open('/workspace/cl/cands/%s__%s.json' % (s, name)))
        labs = json.load(open('/workspace/cl/clab/%s__%s.json' % (s, name)))
        sel = [i for i, (r, l) in enumerate(zip(rows, labs)) if r['fork'] >= 0.2 or l in ('P', 'D')]
        tri = [(rows[i]['p'], rows[i]['a'], rows[i]['b']) for i in sel]
        res = {}
        if tri:
            sub = {n: nodes[n] for t in tri for n in t}
            ids, emb = er.embeddings(name, sub); lookup = {n: i for i, n in enumerate(ids)}
            fg = [fork_geometry(chain(p, prev, pos), chain(a, out, pos), chain(b, out, pos)) for p, a, b in tri]
            pr = er.score('fork', tri, fg, emb, lookup)
            res = {str(i): float(x) for i, x in zip(sel, pr)}
        dst.write_text(json.dumps(res))
        for p in Path('/workspace/cl/ev_cache_b2_%d' % wi).glob('*'): p.unlink()
        print('OK', s, name, len(sel), round(time.time() - t0, 1), flush=True)
    except Exception as e:
        print('ERR', s, name, repr(e)[:200], traceback.format_exc()[-800:], flush=True)
print('WORKER_DONE', wi, flush=True)
