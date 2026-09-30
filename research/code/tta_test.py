"""Fork-head TTA test on labeled DC candidates. usage: tta_test.py <labeled.json> <graph_dir> <out_json>"""
import os, sys, json
from pathlib import Path
from collections import defaultdict
from multiprocessing import get_context
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
ART = '/workspace/art_b56/artifact_bundle'
lab_path, gdir, outp = sys.argv[1], sys.argv[2], sys.argv[3]
DATA = os.environ.get('LIN_DATA', '/workspace/data/train')
FLIPS = ['id', 'x', 'y', 'xy']
def init(gl):
    import multiprocessing as mp
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gl[(mp.current_process()._identity[0] - 1) % len(gl)])
    os.environ['POLARS_MAX_THREADS'] = '2'
def job(args):
    name, rows = args
    sys.path.insert(0, ART)
    import numpy as np, evalx, cell_event
    from refine_events import EventRefiner
    from cell_event import SCALE, chain, fork_geometry
    nodes, edges = evalx.load_graph_json(gdir + '/' + name + '.json')
    out = defaultdict(list); prev = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); out[s].append(d); prev[d] = s
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
    orig = cell_event.Movie.patches
    res = {}
    need = {x for r in rows for x in (r['p'], r['a'], r['b'])}
    sub = {n: nodes[n] for n in need}
    tri = [(r['p'], r['a'], r['b']) for r in rows]
    fg = [fork_geometry(chain(p, prev, pos), chain(a, out, pos), chain(b, out, pos)) for p, a, b in tri]
    for fl in FLIPS:
        def patched(self, times, coords, _fl=fl):
            x = orig(self, times, coords)
            if 'x' in _fl: x = x[..., ::-1]
            if 'y' in _fl: x = x[..., ::-1, :]
            return np.ascontiguousarray(x)
        cell_event.Movie.patches = patched
        er = EventRefiner([Path(ART) / 'b1_best.pt'], DATA, {'fork_ensemble': 'first', 'edge_ensemble': 'last'}, Path('/workspace/cache/tta_' + fl))
        ids, emb = er.embeddings(name, sub); lookup = {n: i for i, n in enumerate(ids)}
        res[fl] = er.score('fork', tri, fg, emb, lookup).tolist()
    cell_event.Movie.patches = orig
    return name, [dict(r, **{'f_' + fl: res[fl][i] for fl in FLIPS}) for i, r in enumerate(rows)]
if __name__ == '__main__':
    import random
    random.seed(0)
    L = json.load(open(lab_path))
    by = defaultdict(list)
    for r in L:
        if r['lab'] == 'pos': by[r['movie']].append(r)
    negs = defaultdict(list)
    for r in L:
        if r['lab'] == 'neg': negs[r['movie']].append(r)
    for m, ns in negs.items():
        random.shuffle(ns); by[m] += ns[:1500]
    with get_context('spawn').Pool(6, initializer=init, initargs=([0, 1],)) as pool: out = pool.map(job, list(by.items()), chunksize=1)
    rows = [r for m, rr in out for r in rr]
    json.dump(rows, open(outp, 'w'))
    import numpy as np
    for typ in ['stolen', 'start']:
        R = [r for r in rows if r['typ'] == typ]
        if not R: continue
        y = np.array([r['lab'] == 'pos' for r in R])
        for fl in FLIPS + ['avg']:
            s = np.array([np.mean([r['f_' + f] for f in FLIPS]) if fl == 'avg' else r['f_' + fl] for r in R])
            rk = (-s).argsort().argsort() + 1
            print(typ, fl, 'npos', int(y.sum()), 'nneg', int((~y).sum()), 'pos ranks', sorted(rk[y].tolist())[:10], 'n>=0.9 pos/neg', int(((s >= 0.9) & y).sum()), int(((s >= 0.9) & ~y).sum()))
