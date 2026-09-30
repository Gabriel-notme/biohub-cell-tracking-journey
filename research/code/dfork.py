"""Double-fork resolution test. usage: dfork.py <graph_dir> <out_json> [K]"""
import os, sys, json, glob
from pathlib import Path
from collections import defaultdict
from multiprocessing import get_context
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
ART = '/workspace/art_b56/artifact_bundle'; P12 = '/workspace/p12ds'
gdir, outp = sys.argv[1], sys.argv[2]; K = int(sys.argv[3]) if len(sys.argv) > 3 else 2
DATA = os.environ.get('LIN_DATA', '/workspace/data/train')
_G = {}
def init(gl):
    import multiprocessing as mp
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gl[(mp.current_process()._identity[0] - 1) % len(gl)])
    os.environ['POLARS_MAX_THREADS'] = '2'; os.environ['BIOHUB_ART'] = ART
def er():
    if 'er' not in _G:
        sys.path.insert(0, ART); sys.path.insert(0, P12)
        from refine_events import EventRefiner
        _G['er'] = EventRefiner([Path(ART) / 'b1_best.pt'], DATA, {'fork_ensemble': 'first', 'edge_ensemble': 'last'}, Path('/workspace/cache/dfork'))
    return _G['er']
def pairs_of(nodes, edges):
    succ = defaultdict(list); par = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s
    forks = [n for n in nodes if len(succ.get(n, [])) >= 2]
    fs = set(forks); pairs = []
    for f in forks:
        fr = [(c, 1, c) for c in succ[f]]
        while fr:
            x, dd, branch = fr.pop()
            if x in fs: pairs.append((f, x, branch, dd))
            if dd < K:
                for y in succ.get(x, []): fr.append((y, dd + 1, branch))
    return pairs, succ, par
def job(p):
    sys.path.insert(0, ART); sys.path.insert(0, P12)
    import numpy as np, evalx
    import tracking_cellmot.division_metrics as DM
    from cell_event import SCALE, chain, fork_geometry
    name = Path(p).stem
    nodes, edges = evalx.load_graph_json(p)
    pairs, succ, par = pairs_of(nodes, edges)
    res = {'movie': name, 'pairs': []}
    variants = {'none': set()}
    if pairs:
        pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
        need = set()
        for f1, f2, br, dd in pairs:
            need |= {f1, f2, *succ[f1], *succ[f2]}
        e = er(); ids, emb = e.embeddings(name, {n: nodes[n] for n in need}); lookup = {n: i for i, n in enumerate(ids)}
        tri = []
        for f1, f2, br, dd in pairs:
            for f in (f1, f2):
                a, b = succ[f][:2]; tri.append((f, a, b))
        fg = [fork_geometry(chain(f, par, pos), chain(a, succ, pos), chain(b, succ, pos)) for f, a, b in tri]
        fp = e.score('fork', tri, fg, emb, lookup)
        pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
        sd = DM.score_divisions(pred, evalx.load_gt(name)[0], scale=evalx.SCALE, max_distance=7.)
        tp = {inv[x] for x in sd.tp_forks}; fpf = {inv[x] for x in sd.fp_forks}
        rm = {'lowprob': set(), 'earlier': set(), 'later': set()}
        for i, (f1, f2, br, dd) in enumerate(pairs):
            p1, p2 = float(fp[2 * i]), float(fp[2 * i + 1])
            other1 = [c for c in succ[f1] if c != br][0]
            d2 = sorted(succ[f2], key=lambda c: -np.linalg.norm(pos[c] - pos[f2]))[0]
            e1, e2 = (f1, other1), (f2, d2)
            res['pairs'].append({'f1': f1, 'f2': f2, 't1': nodes[f1]['t'], 'dt': dd, 'p1': p1, 'p2': p2, 'lab1': 'TP' if f1 in tp else ('FP' if f1 in fpf else 'unl'), 'lab2': 'TP' if f2 in tp else ('FP' if f2 in fpf else 'unl')})
            rm['lowprob'].add(e1 if p1 < p2 else e2); rm['earlier'].add(e1); rm['later'].add(e2)
        variants.update(rm)
    rows = []
    for vn, rmset in variants.items():
        ne = [x for x in edges if (int(x['source_id']), int(x['target_id'])) not in rmset]
        r = evalx.score_movie(name, nodes, ne); r['v'] = vn; rows.append(r)
    for vn in ['lowprob', 'earlier', 'later']:
        if vn not in variants: r = dict(rows[0]); r['v'] = vn; rows.append(r)
    res['rows'] = rows
    return res
if __name__ == '__main__':
    ps = sorted(glob.glob(gdir + '/*.json'))
    with get_context('spawn').Pool(6, initializer=init, initargs=([0, 1],)) as pool: res = pool.map(job, ps, chunksize=1)
    json.dump(res, open(outp, 'w'))
    from tracking_cellmot.metrics import summarise
    for vn in ['none', 'lowprob', 'earlier', 'later']:
        rr = [r for x in res for r in x['rows'] if r['v'] == vn]; s = summarise(rr)
        print('%-8s score %.6f E %.6f div %d/%d/%d' % (vn, s['score'], s['edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn']))
    allp = [dict(pp, movie=x['movie']) for x in res for pp in x['pairs']]
    print('pairs', len(allp))
    for pp in allp:
        if pp['lab1'] != 'unl' or pp['lab2'] != 'unl': print(pp)
    import numpy as np
    print('lowprob-removes-later frac', np.mean([pp['p2'] < pp['p1'] for pp in allp]) if allp else None)
