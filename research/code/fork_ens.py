"""Fork-head ensemble test on labeled DC candidates: b1 vs b2 vs mean. usage: fork_ens.py <labeled.json> <graph_dir> <out_json>"""
import os, sys, json
from pathlib import Path
from collections import defaultdict
from multiprocessing import get_context
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
ART = '/workspace/art_b56/artifact_bundle'
lab_path, gdir, outp = sys.argv[1], sys.argv[2], sys.argv[3]
DATA = os.environ.get('LIN_DATA', '/workspace/data/train')
MODELS = os.environ.get('FE_MODELS', 'b1_best.pt,b2_pretrained_best.pt').split(',')
def init(gl):
    import multiprocessing as mp
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gl[(mp.current_process()._identity[0] - 1) % len(gl)])
    os.environ['POLARS_MAX_THREADS'] = '2'
def job(args):
    name, rows = args
    sys.path.insert(0, ART)
    import numpy as np, evalx
    from refine_events import EventRefiner
    from cell_event import SCALE, chain, fork_geometry
    nodes, edges = evalx.load_graph_json(gdir + '/' + name + '.json')
    out = defaultdict(list); prev = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); out[s].append(d); prev[d] = s
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
    need = {x for r in rows for x in (r['p'], r['a'], r['b'])}
    sub = {n: nodes[n] for n in need}
    tri = [(r['p'], r['a'], r['b']) for r in rows]
    fg = [fork_geometry(chain(p, prev, pos), chain(a, out, pos), chain(b, out, pos)) for p, a, b in tri]
    res = {}
    for m in MODELS:
        er = EventRefiner([Path(ART) / m], DATA, {'fork_ensemble': 'first', 'edge_ensemble': 'last'}, Path('/workspace/cache/fe_' + m.split('.')[0]))
        ids, emb = er.embeddings(name, sub); lookup = {n: i for i, n in enumerate(ids)}
        res[m] = er.score('fork', tri, fg, emb, lookup).tolist()
    return name, [dict(r, **{'f_' + m.split('_')[0]: res[m][i] for m in MODELS}) for i, r in enumerate(rows)]
if __name__ == '__main__':
    import random
    random.seed(0)
    L = json.load(open(lab_path))
    by = defaultdict(list); negs = defaultdict(list)
    for r in L:
        if r['lab'] == 'pos': by[r['movie']].append(r)
        elif r['lab'] == 'neg': negs[r['movie']].append(r)
    del L
    for m, ns in negs.items():
        ns.sort(key=lambda r: -r['fork'])
        top = ns[:300]; rest = ns[300:]; random.shuffle(rest)
        by[m] += top + rest[:1200]
    with get_context('spawn').Pool(6, initializer=init, initargs=([0, 1],)) as pool: out = pool.map(job, list(by.items()), chunksize=1)
    rows = [r for m, rr in out for r in rr]
    json.dump(rows, open(outp, 'w'))
    import numpy as np
    keys = ['f_' + m.split('_')[0] for m in MODELS]
    lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    for typ in ['stolen', 'start']:
        R = [r for r in rows if r['typ'] == typ]
        if not R: continue
        y = np.array([r['lab'] == 'pos' for r in R])
        S = {k: np.array([r[k] for r in R]) for k in keys}
        if len(keys) > 1:
            S['mean'] = 1 / (1 + np.exp(-np.mean([lg(S[k]) for k in keys], 0)))
            S['min'] = np.min([S[k] for k in keys], 0)
        for k, s in S.items():
            rk = (-s).argsort().argsort() + 1
            msg = ' '.join('>=%.2f:%d/%d' % (th, int(((s >= th) & y).sum()), int(((s >= th) & ~y).sum())) for th in [0.5, 0.8, 0.9, 0.95, 0.97, 0.99])
            print(typ, k, 'npos', int(y.sum()), 'nneg', int((~y).sum()), 'pos ranks', sorted(rk[y].tolist())[:12], msg)
