"""Explore: transfer-verifier scores for DSR candidates (b1 >= 0.9) with GT labels.
usage: dsr_verify.py <graph_dir> <full_dir> <out_json>"""
import os, sys, json, glob
from pathlib import Path
from collections import defaultdict
from multiprocessing import get_context
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
ART = '/workspace/art_b56/artifact_bundle'; P12 = '/workspace/p12ds'
gdir, fdir, outp = sys.argv[1], sys.argv[2], sys.argv[3]
DATA = os.environ.get('LIN_DATA', '/workspace/data/train')
_G = {}
def init(gl):
    import multiprocessing as mp
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gl[(mp.current_process()._identity[0] - 1) % len(gl)])
    os.environ['POLARS_MAX_THREADS'] = '2'; os.environ['BIOHUB_ART'] = ART
def models():
    if 'er' not in _G:
        from refine_events import EventRefiner
        from transfer_model import load_transfer
        _G['er'] = EventRefiner([Path(ART) / 'b1_best.pt'], DATA, {'fork_ensemble': 'first', 'edge_ensemble': 'last'}, Path('/workspace/cache/dsrv'))
        _G['ver'], _ = load_transfer(Path(ART) / 'transfer_v2_frozen.pt')
    return _G['er'], _G['ver']
def job(p):
    sys.path.insert(0, ART); sys.path.insert(0, P12)
    import numpy as np, torch, zarr, evalx
    from scipy.spatial import cKDTree
    from cell_event import SCALE, chain, fork_geometry, Movie
    from tracking_cellmot.metrics import evaluate
    import dsr
    K = evalx.K
    name = Path(p).stem
    nodes, edges = evalx.load_graph_json(p)
    out = defaultdict(list); prev = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); out[s].append(d); prev[d] = s
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    trees = {t: cKDTree(np.array([pos[n] for n in ns])) for t, ns in frames.items()}
    FT, FV = dsr.load_full(fdir + '/%s.geff' % name); FP = FV * SCALE
    dropped = {}
    for t in np.unique(FT):
        idx = np.where(FT == t)[0]
        if int(t) in trees:
            d, _ = trees[int(t)].query(FP[idx]); idx = idx[d > 4.0]
        if len(idx): dropped[int(t)] = idx
    dtrees = {t: cKDTree(FP[ix]) for t, ix in dropped.items()}
    cands = []
    for t, ns in frames.items():
        if t + 1 not in dtrees: continue
        ix = dropped[t + 1]
        for pn in ns:
            ch = out.get(pn, [])
            if len(ch) != 1: continue
            a = ch[0]
            for j in dtrees[t + 1].query_ball_point(pos[pn], 13.0):
                fi = int(ix[j])
                if np.linalg.norm(FP[fi] - pos[a]) > 20.0: continue
                cands.append((pn, a, fi))
    if not cands: return []
    er, ver = models()
    NEW0 = 10 ** 9; newid = {}
    for pn, a, fi in cands:
        if fi not in newid: newid[fi] = NEW0 + len(newid)
    sub = {n: nodes[n] for c in cands for n in c[:2]}
    for fi, nid in newid.items(): sub[nid] = {'t': int(FT[fi]), 'z': float(FV[fi][0]), 'y': float(FV[fi][1]), 'x': float(FV[fi][2])}
    ids, emb = er.embeddings(name, sub); lookup = {n: i for i, n in enumerate(ids)}
    tri = [(pn, a, newid[fi]) for pn, a, fi in cands]
    fg = [fork_geometry(chain(pn, prev, pos), chain(a, out, pos), FP[fi][None]) for pn, a, fi in cands]
    pb1 = er.score('fork', tri, fg, emb, lookup)
    keep = [i for i in range(len(cands)) if pb1[i] >= 0.9]
    if not keep: return []
    # GT labels
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x]) * np.array([1.625, .40625, .40625]) for i, z, y, x in zip(*[ga[c].to_list() for c in [K.NODE_ID, 'z', 'y', 'x']])}
    ea = gt.edge_attrs(); gs = defaultdict(list)
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(s)].append(int(d))
    # verifier
    movie = Movie(Path(DATA) / (name + '.zarr'), context=7)
    reqs = []; rix = {}
    def req(t, c):
        key = (int(t), *np.rint(np.asarray(c) * 32).astype(int).tolist())
        if key not in rix: rix[key] = len(reqs); reqs.append((int(t), np.asarray(c)))
        return rix[key]
    rows = []; geo = []
    for i in keep:
        pn, a, fi = cands[i]; t = int(nodes[pn]['t'])
        rows.append([req(t, pos[pn] / SCALE), req(t + 1, pos[a] / SCALE), req(t + 1, FV[fi])]); geo.append(fg[i])
    bank = torch.empty((len(reqs), 3, 64), device='cuda', dtype=torch.float32)
    order = sorted(range(len(reqs)), key=lambda k: reqs[k][0])
    with torch.inference_mode():
        for st in range(0, len(order), 192):
            ii = order[st:st + 192]; x = torch.from_numpy(movie.patches([reqs[k][0] for k in ii], [reqs[k][1] for k in ii])).cuda().float()
            with torch.autocast('cuda', dtype=torch.float16): z = ver.base.encode(torch.cat([x[:, 0:3], x[:, 2:5], x[:, 4:7]], 0))
            bank[ii] = z.reshape(3, len(ii), 64).transpose(0, 1).float()
        ix = torch.tensor(rows, device='cuda'); g = torch.from_numpy(np.stack(geo)).cuda()
        with torch.autocast('cuda', dtype=torch.float16): pv = ver(bank[ix], g, 'fork')
        pv = pv.float().sigmoid().cpu().numpy()
    res = []
    for k, i in enumerate(keep):
        pn, a, fi = cands[i]; g0 = p2g.get(pn); lab = 'unl'
        if g0 is not None and gs.get(g0):
            kids = gs[g0]
            if len(kids) == 2 and p2g.get(a) in kids:
                other = [c for c in kids if c != p2g.get(a)][0]
                lab = 'pos' if np.linalg.norm(gpos[other] - FP[fi]) <= 7 else 'neg'
            else: lab = 'neg'
        res.append({'movie': name, 'b1': float(pb1[i]), 'ver': float(pv[k]), 'lab': lab})
    return res
if __name__ == '__main__':
    ps = sorted(glob.glob(gdir + '/*.json'))
    with get_context('spawn').Pool(6, initializer=init, initargs=([0, 1],)) as pool: res = [r for rr in pool.map(job, ps, chunksize=1) for r in rr]
    json.dump(res, open(outp, 'w'))
    for r in res:
        if r['lab'] != 'unl': print(r)
    import numpy as np
    u = [r['ver'] for r in res if r['lab'] == 'unl']
    print('unl n', len(u), 'ver quantiles', np.round(np.quantile(u, [.1, .25, .5, .75, .9]), 3).tolist() if u else None)
