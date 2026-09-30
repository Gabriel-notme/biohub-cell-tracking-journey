"""Fork timing shift test: b1 score of original fork (F->A,B) vs fork moved one frame earlier (P->F, P->X, X->B).
usage: fts_test.py <graph_dir> <fullgraph_dir> <tdiv.json> <out.json>"""
import os, sys, json
from pathlib import Path
from collections import defaultdict
from multiprocessing import get_context
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/p12ds')
ART = '/workspace/art_b56/artifact_bundle'
gdir, fdir, tdiv_path, outp = sys.argv[1:5]
DATA = os.environ.get('LIN_DATA', '/workspace/data/train')
def init(gl):
    import multiprocessing as mp
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gl[(mp.current_process()._identity[0] - 1) % len(gl)])
    os.environ['POLARS_MAX_THREADS'] = '2'
def job(args):
    name, lab = args
    sys.path.insert(0, ART)
    import numpy as np, evalx, dsr
    from refine_events import EventRefiner
    from cell_event import SCALE, chain, fork_geometry
    from scipy.spatial import cKDTree
    lab = {int(k): v for k, v in lab.items()}
    nodes, edges = evalx.load_graph_json(gdir + '/' + name + '.json')
    out = defaultdict(list); prev = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); out[s].append(d); prev[d] = s
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    trees = {t: cKDTree(np.array([pos[n] for n in ns])) for t, ns in frames.items()}
    FT, FV = dsr.load_full(fdir + '/' + name + '.geff'); FPs = FV * SCALE
    dropped = {}
    for t in np.unique(FT):
        idx = np.where(FT == t)[0]
        if int(t) in trees:
            d, _ = trees[int(t)].query(FPs[idx]); idx = idx[d > 2.0]
        if len(idx): dropped[int(t)] = idx
    dtrees = {t: cKDTree(FPs[ix]) for t, ix in dropped.items()}
    forks = [n for n in nodes if len(out.get(n, [])) == 2]
    rows = []; sub = {}; NEW = 10 ** 9; k = 0
    for F in forks:
        P = prev.get(F)
        if P is None or len(out.get(P, [])) != 1: continue
        A, B = out[F]
        if np.linalg.norm(pos[B] - pos[F]) < np.linalg.norm(pos[A] - pos[F]): A, B = B, A
        t = int(nodes[F]['t'])
        mid = (pos[F] + pos[B]) / 2; X = mid; src = 'mid'
        if t in dtrees:
            d, j = dtrees[t].query(mid)
            if d <= 3.0: X = FPs[dropped[t][j]]; src = 'drop'
        xid = NEW + k; k += 1
        Xr = X / SCALE
        sub[xid] = {'t': t, 'z': float(Xr[0]), 'y': float(Xr[1]), 'x': float(Xr[2])}
        pos[xid] = np.asarray(X, np.float32)
        for n in (P, F, A, B): sub[n] = nodes[n]
        rows.append(dict(movie=name, F=int(F), P=int(P), A=int(A), B=int(B), X=xid, t=t, src=src, lab=lab.get(int(F), 'unl'),
                         dAB=float(np.linalg.norm(pos[A] - pos[B])), dFB=float(np.linalg.norm(pos[F] - pos[B])), dFA=float(np.linalg.norm(pos[F] - pos[A])),
                         dPF=float(np.linalg.norm(pos[P] - pos[F])), dXF=float(np.linalg.norm(pos[xid] - pos[F]))))
    if not rows: return []
    er = EventRefiner([Path(ART) / 'b1_best.pt'], DATA, {'fork_ensemble': 'first', 'edge_ensemble': 'last'}, Path('/workspace/cache/fts'))
    ids, emb = er.embeddings(name, sub); lookup = {n: i for i, n in enumerate(ids)}
    tri0 = [(r['F'], r['A'], r['B']) for r in rows]
    fg0 = [fork_geometry(chain(r['F'], prev, pos), chain(r['A'], out, pos), chain(r['B'], out, pos)) for r in rows]
    s0 = er.score('fork', tri0, fg0, emb, lookup)
    tri1 = []; fg1 = []
    for r in rows:
        o2 = dict(out); o2[r['F']] = [r['A']]; o2[r['X']] = [r['B']]
        tri1.append((r['P'], r['F'], r['X']))
        fg1.append(fork_geometry(chain(r['P'], prev, pos), chain(r['F'], o2, pos), chain(r['X'], o2, pos)))
    s1 = er.score('fork', tri1, fg1, emb, lookup)
    for r, a, b in zip(rows, s0, s1): r['s0'] = float(a); r['s1'] = float(b)
    return rows
if __name__ == '__main__':
    import glob
    T = json.load(open(tdiv_path))
    lab = defaultdict(dict)
    for r in T:
        if r['kind'] == 'TP':
            for off, kind, fid in r['near_forks']:
                if kind == 'tp': lab[r['movie']][fid] = 'tp%+d' % off
        if r['kind'] == 'FP':
            ng = r['near_gt_div']
            lab[r['movie']][r['p']] = ('fp_gt%+d' % min(ng, key=abs)) if ng else 'fp'
    names = sorted(Path(p).stem for p in glob.glob(gdir + '/*.json'))
    with get_context('spawn').Pool(6, initializer=init, initargs=([0, 1],)) as pool:
        res = pool.map(job, [(n, {str(k): v for k, v in lab[n].items()}) for n in names], chunksize=1)
    rows = [r for x in res for r in x]
    json.dump(rows, open(outp, 'w'))
    import numpy as np
    from collections import Counter
    print('forks', len(rows), Counter(r['lab'] for r in rows))
    for r in sorted(rows, key=lambda r: r['lab']):
        if r['lab'] != 'unl':
            print('%-10s %s F%d t%d s0 %.3f s1 %.3f src %s dAB %.1f dFB %.1f dFA %.1f dXF %.1f' % (r['lab'], r['movie'], r['F'], r['t'], r['s0'], r['s1'], r['src'], r['dAB'], r['dFB'], r['dFA'], r['dXF']))
    U = [r for r in rows if r['lab'] == 'unl']
    s0 = np.array([r['s0'] for r in U]); s1 = np.array([r['s1'] for r in U])
    print('unl n', len(U), 'frac s1>s0', float((s1 > s0).mean()), 'frac s1>=0.9', float((s1 >= .9).mean()), 'frac s1>s0+0.2', float((s1 > s0 + .2).mean()))
