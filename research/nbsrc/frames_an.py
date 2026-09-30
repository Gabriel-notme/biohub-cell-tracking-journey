"""Per transition t->t+1 of every movie: exact-identical frames, global shift (3D phase correlation on a 4x-downsampled volume, um),
GT median displacement (um) and GT edge count, and P20 (H100 rerun graphs) FN edges among GT edges whose both ends are matched.
Writes /workspace/nbrun/frames_an.json and prints a summary by transition class."""
import os, sys, json, glob
for v in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'): os.environ[v] = '1'
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
LST = {'hold36': 'hold36', 'prev4': 'preview4', 'audit32': 'audit32', 't127a': 't127a', 't127b': 't127b'}
mset = {m: s for s, l in LST.items() for m in open('/workspace/%s.txt' % l).read().split()}

def pc_shift(a, b):
    A = np.fft.fftn(a); B = np.fft.fftn(b)
    R = A * np.conj(B); R /= np.abs(R) + 1e-9
    r = np.real(np.fft.ifftn(R)); i = np.array(np.unravel_index(np.argmax(r), r.shape))
    sh = np.where(i > np.array(r.shape) // 2, i - np.array(r.shape), i)
    return sh

def job(m):
    import zarr, evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    arr = zarr.open_group('/workspace/data/train/%s.zarr' % m, mode='r')['0']
    T = arr.shape[0]
    prev = np.asarray(arr[0]); ident = []; shifts = []
    for t in range(T - 1):
        cur = np.asarray(arr[t + 1])
        ident.append(bool(np.array_equal(prev, cur)))
        a = prev[:, ::4, ::4].astype(np.float32); b = cur[:, ::4, ::4].astype(np.float32)
        a -= a.mean(); b -= b.mean()
        sh = pc_shift(b, a)  # shift of frame t+1 relative to t (downsampled voxels)
        shifts.append((sh * np.array([1, 4, 4]) * S).tolist())
        prev = cur
    s = mset[m]
    f = '/workspace/cl/p21/ps_p20ref_%s/graphs/%s.json' % (s, m)
    nodes, edges = evalx.load_graph_json(f)
    g, mp = evalx.to_graph(nodes, edges, rounding=True)
    gt, _ = evalx.load_gt(m)
    evaluate(g, gt, scale=tuple(S), max_distance=7.0)
    na = g.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID]).to_dicts()
    g2p = {int(r[K.MATCHED_NODE_ID]): int(r[K.NODE_ID]) for r in na if r[K.MATCHED_NODE_ID] is not None and int(r[K.MATCHED_NODE_ID]) != -1}
    pe = set()
    ea = g.edge_attrs().to_dicts()
    for r in ea: pe.add((int(r['source_id']), int(r['target_id'])))
    ga = {r[K.NODE_ID]: r for r in gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x']).to_dicts()}
    per = {t: {'gt_edges': 0, 'both_matched': 0, 'fn': 0, 'unmatched_end': 0, 'disp': []} for t in range(T - 1)}
    for r in gt.edge_attrs().to_dicts():
        a, b = int(r['source_id']), int(r['target_id']); t = int(ga[a]['t'])
        if int(ga[b]['t']) != t + 1: continue
        d = per[t]; d['gt_edges'] += 1
        d['disp'].append((np.array([ga[b][c] - ga[a][c] for c in 'zyx']) * S).tolist())
        if a in g2p and b in g2p:
            d['both_matched'] += 1
            if (g2p[a], g2p[b]) not in pe: d['fn'] += 1
        else: d['unmatched_end'] += 1
    out = []
    for t in range(T - 1):
        d = per[t]; disp = np.array(d['disp']) if d['disp'] else np.zeros((0, 3))
        out.append(dict(t=t, ident=ident[t], shift=shifts[t], gt_edges=d['gt_edges'], both=d['both_matched'], fn=d['fn'], unm=d['unmatched_end'],
                        gt_med=np.median(disp, 0).tolist() if len(disp) else None))
    return dict(movie=m, set=s, rows=out)

if __name__ == '__main__':
    ms = sorted(mset)
    with Pool(40) as p: R = p.map(job, ms, chunksize=1)
    json.dump(R, open('/workspace/nbrun/frames_an.json', 'w'))
    rows = [(r['movie'], x) for r in R for x in r['rows']]
    def cls(x):
        sh = np.linalg.norm(x['shift'])
        if x['ident']: return 'identical'
        if sh >= 3.0: return 'jump>=3um'
        if sh >= 1.5: return 'jump1.5-3um'
        return 'normal'
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0, 0, 0, set()])
    for m, x in rows:
        c = cls(x); a = agg[c]; a[0] += 1; a[1] += x['gt_edges']; a[2] += x['both']; a[3] += x['fn']; a[4] += x['unm']; a[5].add(m)
    print('class        transitions movies gt_edges both_matched FN(rate) unmatched_end(rate)')
    for c, a in sorted(agg.items()):
        print('%-12s %6d %6d %8d %8d %6d (%.4f) %6d (%.4f)' % (c, a[0], len(a[5]), a[1], a[2], a[3], a[3] / max(1, a[2]), a[4], a[4] / max(1, a[1])))
    ident_m = sorted({m for m, x in rows if x['ident']}); print('movies with identical consecutive frames:', len(ident_m), ident_m[:12])
    gm = [x['gt_med'] for m, x in rows if x['ident'] and x['gt_med'] is not None]
    if gm: print('GT median displacement across identical frames (um), mean over transitions:', np.round(np.mean(np.abs(np.array(gm)), 0), 3).tolist(), 'n', len(gm))
    big = sorted([(np.linalg.norm(x['shift']), m, x['t'], x['fn'], x['both']) for m, x in rows], reverse=True)[:12]
    print('largest shifts (um, movie, t, FN, both):', [(round(a, 1), b, c, d, e) for a, b, c, d, e in big])
