"""Dropped-sister recovery (DSR) test: add a second daughter from ILP-dropped detections.
usage: dsr_test.py <graph_dir> <full_dir> <out_json> [ths]"""
import os, sys, json, glob
from pathlib import Path
from collections import defaultdict
from multiprocessing import get_context
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
ART = '/workspace/art_b56/artifact_bundle'; P12 = '/workspace/p12ds'
gdir, fdir, outp = sys.argv[1], sys.argv[2], sys.argv[3]
THS = [float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else '0.5,0.7,0.8,0.9,0.95,0.98').split(',')]
DATA = os.environ.get('LIN_DATA', '/workspace/data/train')
MAX_PB, MAX_AB, SEP, FUT = 13.0, 20.0, 4.0, 6.0
DETAIL_TH = float(os.environ.get('DSR_DETAIL_TH', '0.9'))
_G = {}
def init(gl):
    import multiprocessing as mp
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gl[(mp.current_process()._identity[0] - 1) % len(gl)])
    os.environ['POLARS_MAX_THREADS'] = '2'; os.environ['BIOHUB_ART'] = ART
def er():
    if 'er' not in _G:
        from refine_events import EventRefiner
        _G['er'] = EventRefiner([Path(ART) / 'b1_best.pt'], DATA, {'fork_ensemble': 'first', 'edge_ensemble': 'last'}, Path('/workspace/cache/dsr'))
    return _G['er']
def job(p):
    sys.path.insert(0, ART); sys.path.insert(0, P12)
    import numpy as np, zarr, evalx
    from scipy.spatial import cKDTree
    from cell_event import SCALE, chain, fork_geometry
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
    FV = np.stack([np.asarray(g['nodes/props/%s/values' % k][:]) for k in 'zyx'], 1).astype(np.float32)
    FP = FV * SCALE
    dropped = defaultdict(list)
    for t in np.unique(FT):
        idx = np.where(FT == t)[0]
        if t in trees:
            d, _ = trees[t].query(FP[idx]); idx = idx[d > SEP]
        dropped[int(t)] = idx.tolist()
    dtrees = {t: cKDTree(FP[ix]) for t, ix in dropped.items() if ix}
    starts = defaultdict(list)
    for n, v in nodes.items():
        if n not in prev: starts[int(v['t'])].append(n)
    stree = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in starts.items() if ns}
    cands = []
    for t, ns in frames.items():
        if t + 1 not in dtrees: continue
        ix = dropped[t + 1]
        for pn in ns:
            ch = out.get(pn, [])
            if len(ch) != 1: continue
            a = ch[0]
            for j in dtrees[t + 1].query_ball_point(pos[pn], MAX_PB):
                fi = ix[j]
                if np.linalg.norm(FP[fi] - pos[a]) > MAX_AB: continue
                cands.append((pn, a, fi))
    res = {'movie': name, 'n_cands': len(cands)}
    base_row = evalx.score_movie(name, nodes, edges); base_row['th'] = 'base'
    rows = [base_row]
    if cands:
        NEW0 = 10 ** 9
        newnodes = {}
        for k, (pn, a, fi) in enumerate(cands):
            if fi not in newnodes: newnodes[fi] = NEW0 + len(newnodes)
        sub = {n: nodes[n] for c in cands for n in c[:2]}
        for fi, nid in newnodes.items():
            sub[nid] = {'t': int(FT[fi]), 'z': float(FV[fi][0]), 'y': float(FV[fi][1]), 'x': float(FV[fi][2])}
        e = er(); ids, emb = e.embeddings(name, sub); lookup = {n: i for i, n in enumerate(ids)}
        fut = {}
        for fi, nid in newnodes.items():
            t2 = int(FT[fi]) + 1; f = None
            if t2 in stree:
                ns2, tr2 = stree[t2]; d, j = tr2.query(FP[fi])
                if d <= FUT: f = ns2[j]
            fut[fi] = f
        tri = [(pn, a, newnodes[fi]) for pn, a, fi in cands]
        fg = []
        for pn, a, fi in cands:
            bch = np.vstack([FP[fi][None], chain(fut[fi], out, pos)]) if fut[fi] is not None else FP[fi][None]
            fg.append(fork_geometry(chain(pn, prev, pos), chain(a, out, pos), bch))
        fp = e.score('fork', tri, fg, emb, lookup)
        order = np.argsort(-fp)
        res['probs'] = sorted([float(x) for x in fp], reverse=True)[:20]
        for th in THS:
            used_p, used_f, used_s = set(), set(), set(); nn = dict(nodes); ne = list(edges); nid = max(nodes) + 1; added = 0
            for i in order:
                if fp[i] < th: break
                pn, a, fi = cands[i]
                if pn in used_p or fi in used_f: continue
                if fut[fi] is not None and fut[fi] in used_s: continue
                used_p.add(pn); used_f.add(fi)
                nn[nid] = {'node_id': nid, 't': int(FT[fi]), 'z': float(FV[fi][0]), 'y': float(FV[fi][1]), 'x': float(FV[fi][2])}
                ne.append({'source_id': pn, 'target_id': nid, 'dsr': 1})
                if fut[fi] is not None: ne.append({'source_id': nid, 'target_id': fut[fi], 'dsr': 1}); used_s.add(fut[fi])
                nid += 1; added += 1
            r = evalx.score_movie(name, nn, ne); r['th'] = th; r['added'] = added; rows.append(r)
            if abs(th - DETAIL_TH) < 1e-9 and added:
                import tracking_cellmot.division_metrics as DM
                pred, mapping = evalx.to_graph(nn, ne); inv = {v: k for k, v in mapping.items()}
                sd = DM.score_divisions(pred, evalx.load_gt(name)[0], scale=evalx.SCALE, max_distance=7.)
                tpf = {inv[x] for x in sd.tp_forks}; fpf = {inv[x] for x in sd.fp_forks}
                det = []
                for i in order:
                    pn, a, fi = cands[i]
                    if pn in used_p and fp[i] >= th:
                        det.append({'p': pn, 'prob': float(fp[i]), 'fut': fut[fi] is not None, 'lab': 'TP' if pn in tpf else ('FP' if pn in fpf else 'unl'), 'hist': len(chain(pn, prev, pos)), 'dpb': float(np.linalg.norm(FP[fi] - pos[pn])), 'dab': float(np.linalg.norm(FP[fi] - pos[a]))})
                res['detail'] = det
    for th in THS:
        if not any(r['th'] == th for r in rows): r = dict(base_row); r['th'] = th; r['added'] = 0; rows.append(r)
    res['rows'] = rows
    return res
if __name__ == '__main__':
    ps = sorted(glob.glob(gdir + '/*.json'))
    with get_context('spawn').Pool(6, initializer=init, initargs=([0, 1],)) as pool: res = pool.map(job, ps, chunksize=1)
    json.dump(res, open(outp, 'w'))
    from tracking_cellmot.metrics import summarise
    for th in ['base'] + THS:
        rr = [r for x in res for r in x['rows'] if r['th'] == th]; s = summarise(rr)
        print('th=%s score %.6f adjE %.6f E %.6f div %d/%d/%d added %d nodes %d' % (th, s['score'], s['adj_edge_jaccard'], s['edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn'], sum(r.get('added', 0) for r in rr), sum(r['num_pred_nodes'] for r in rr)), flush=True)
    print('cands', sum(x['n_cands'] for x in res))
    import collections
    det = [d for x in res for d in x.get('detail', [])]
    for d in det:
        if d['lab'] != 'unl': print('DETAIL', d)
    print('fut frac by lab', {l: (sum(d['fut'] for d in det if d['lab'] == l), sum(1 for d in det if d['lab'] == l)) for l in ['TP', 'FP', 'unl']})
