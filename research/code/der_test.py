"""Dropped-endpoint recovery (DER) test: extend track ends with ILP-dropped detections (optionally bridging to a track start).
usage: der_test.py <graph_dir> <full_dir> <out_json> [ths]"""
import os, sys, json, glob
from pathlib import Path
from collections import defaultdict
from multiprocessing import get_context
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
ART = '/workspace/art_b56/artifact_bundle'; P12 = '/workspace/p12ds'
gdir, fdir, outp = sys.argv[1], sys.argv[2], sys.argv[3]
THS = [float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else '0.8,0.9,0.95,0.98,0.99').split(',')]
MODE = os.environ.get('DER_MODE', 'all')  # all | bridge
DATA = os.environ.get('LIN_DATA', '/workspace/data/train')
R, SEP, FUT = 8.0, 4.0, 6.0
_G = {}
def init(gl):
    import multiprocessing as mp
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gl[(mp.current_process()._identity[0] - 1) % len(gl)])
    os.environ['POLARS_MAX_THREADS'] = '2'; os.environ['BIOHUB_ART'] = ART
def er():
    if 'er' not in _G:
        from refine_events import EventRefiner
        _G['er'] = EventRefiner([Path(ART) / 'b1_best.pt', Path(ART) / 'b2_pretrained_best.pt'], DATA, {'fork_ensemble': 'first', 'edge_ensemble': 'last'}, Path('/workspace/cache/der'))
    return _G['er']
def job(p):
    sys.path.insert(0, ART); sys.path.insert(0, P12)
    import numpy as np, zarr, evalx
    from scipy.spatial import cKDTree
    from cell_event import SCALE, chain, edge_geometry
    name = Path(p).stem
    nodes, edges = evalx.load_graph_json(p)
    out = defaultdict(list); prev = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); out[s].append(d); prev[d] = s
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    trees = {t: cKDTree(np.array([pos[n] for n in ns])) for t, ns in frames.items()}
    g = zarr.open_group(fdir + '/%s.geff' % name, mode='r')
    FT = np.asarray(g['nodes/props/t/values'][:]).astype(int)
    FV = np.stack([np.asarray(g['nodes/props/%s/values' % k][:]) for k in 'zyx'], 1).astype(np.float32); FP = FV * SCALE
    dropped = {}
    for t in np.unique(FT):
        idx = np.where(FT == t)[0]
        if int(t) in trees:
            d, _ = trees[int(t)].query(FP[idx]); idx = idx[d > SEP]
        if len(idx): dropped[int(t)] = idx
    dtrees = {t: cKDTree(FP[ix]) for t, ix in dropped.items()}
    starts = defaultdict(list)
    for n, v in nodes.items():
        if n not in prev: starts[int(v['t'])].append(n)
    stree = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in starts.items() if ns}
    cands = []
    T = max(frames)
    for t, ns in frames.items():
        if t + 1 not in dtrees: continue
        ix = dropped[t + 1]
        for u in ns:
            if out.get(u) or u not in prev: continue
            h = chain(u, prev, pos); v = (h[0] - h[-1]) / max(1, len(h) - 1)
            exp = pos[u] + v
            for j in dtrees[t + 1].query_ball_point(exp, R):
                fi = int(ix[j])
                f = None
                if t + 2 in stree:
                    ns2, tr2 = stree[t + 2]; dd, jj = tr2.query(FP[fi])
                    if dd <= FUT: f = ns2[int(jj)]
                if MODE == 'bridge' and f is None: continue
                cands.append((u, fi, f))
    res = {'movie': name, 'n_cands': len(cands)}
    base = evalx.score_movie(name, nodes, edges); base['th'] = 'base'; rows = [base]
    if cands:
        NEW0 = 10 ** 9; newid = {}
        for u, fi, f in cands:
            if fi not in newid: newid[fi] = NEW0 + len(newid)
        sub = {u: nodes[u] for u, fi, f in cands}
        for u, fi, f in cands:
            if f is not None: sub[f] = nodes[f]
        for fi, nid in newid.items(): sub[nid] = {'t': int(FT[fi]), 'z': float(FV[fi][0]), 'y': float(FV[fi][1]), 'x': float(FV[fi][2])}
        e = er(); ids, emb = e.embeddings(name, sub); lookup = {n: i for i, n in enumerate(ids)}
        rows_e = [(u, newid[fi]) for u, fi, f in cands]
        geo = []
        for u, fi, f in cands:
            fut = np.vstack([FP[fi][None], chain(f, out, pos)]) if f is not None else FP[fi][None]
            geo.append(edge_geometry(chain(u, prev, pos), fut))
        pe = e.score('edge', rows_e, geo, emb, lookup)
        pb = np.ones(len(cands))
        bi = [i for i, (u, fi, f) in enumerate(cands) if f is not None]
        if bi:
            rows_b = [(newid[cands[i][1]], cands[i][2]) for i in bi]
            geo_b = [edge_geometry(np.vstack([FP[cands[i][1]][None], chain(cands[i][0], prev, pos)]), chain(cands[i][2], out, pos)) for i in bi]
            pbv = e.score('edge', rows_b, geo_b, emb, lookup)
            for k, i in enumerate(bi): pb[i] = pbv[k]
        sc = pe * np.where([f is not None for u, fi, f in cands], pb, 1.0)
        order = np.argsort(-sc)
        for th in THS:
            used_u, used_f, used_s = set(), set(), set(); nn = dict(nodes); ne = list(edges); nid = max(nodes) + 1; added = 0; bridges = 0
            for i in order:
                if sc[i] < th: break
                u, fi, f = cands[i]
                if u in used_u or fi in used_f: continue
                if f is not None and (f in used_s or pb[i] < th): f = None
                if MODE == 'bridge' and f is None: continue
                used_u.add(u); used_f.add(fi)
                nn[nid] = {'node_id': nid, 't': int(FT[fi]), 'z': float(FV[fi][0]), 'y': float(FV[fi][1]), 'x': float(FV[fi][2])}
                ne.append({'source_id': u, 'target_id': nid, 'der': 1})
                if f is not None: ne.append({'source_id': nid, 'target_id': f, 'der': 1}); used_s.add(f); bridges += 1
                nid += 1; added += 1
            r = evalx.score_movie(name, nn, ne); r['th'] = th; r['added'] = added; r['bridges'] = bridges; rows.append(r)
    for th in THS:
        if not any(r['th'] == th for r in rows): r = dict(base); r['th'] = th; r['added'] = 0; r['bridges'] = 0; rows.append(r)
    res['rows'] = rows
    return res
if __name__ == '__main__':
    ps = sorted(glob.glob(gdir + '/*.json'))
    with get_context('spawn').Pool(6, initializer=init, initargs=([0, 1],)) as pool: res = pool.map(job, ps, chunksize=1)
    json.dump(res, open(outp, 'w'))
    from tracking_cellmot.metrics import summarise
    for th in ['base'] + THS:
        rr = [r for x in res for r in x['rows'] if r['th'] == th]; s = summarise(rr)
        print('mode=%s th=%s score %.6f adjE %.6f E %.6f div %d/%d/%d added %d bridges %d nodes %d' % (MODE, th, s['score'], s['adj_edge_jaccard'], s['edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn'], sum(r.get('added', 0) for r in rr), sum(r.get('bridges', 0) for r in rr), sum(r['num_pred_nodes'] for r in rr)), flush=True)
    print('cands', sum(x['n_cands'] for x in res))
