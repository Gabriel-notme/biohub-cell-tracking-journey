"""Junk-track pruning. Every predicted node costs 0.1*J/N_total in the adjusted edge Jaccard, so whole isolated tracks
(no parent, no division, i.e. complete components without forks) that are unlikely to carry annotated edges are removed.
A Poisson GBM predicts E[#TP edges] of a track from prediction-only features; a track is removed when
E[TP] < lam * len * be, be = 0.1 * TP_REF / ntot_hat / m_hat (test-time estimate of the per-node break-even)."""
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
S = np.array([1.625, 0.40625, 0.40625])
FEATS = ['len', 't0', 't1', 'z', 'y', 'x', 'sz', 'sy', 'sx', 'v', 'vmax', 'ep_mean', 'ep_min', 'zr', 'rr', 'lrank',
         'dup5', 'dup7', 'nn_med', 'dens10', 'npred', 'ntrk', 'frac_iso']


def tracks(nodes, edges):
    ch = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); ch[a].append(b); par[b] = a
    ep = {}
    for e in edges:
        p = e.get('edge_prob'); ep[int(e['target_id'])] = float(p) if p is not None else np.nan
    out = []
    for n in nodes:
        if n in par: continue
        chain = [n]; ok = True
        while True:
            c = ch.get(chain[-1], [])
            if len(c) == 0: break
            if len(c) > 1: ok = False; break
            chain.append(c[0])
        if ok: out.append(chain)
    return out, ep


def features(nodes, edges):
    trk, ep = tracks(nodes, edges)
    if not trk: return trk, np.zeros((0, len(FEATS)))
    pos = {n: np.array([v['z'], v['y'], v['x']], float) * S for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    trees = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in frames.items()}
    zs = np.array([v['z'] for v in nodes.values()]); zmin, zptp = zs.min(), max(1e-6, np.ptp(zs))
    npred = len(nodes); lens = np.array([len(c) for c in trk])
    rows = []
    for c in trk:
        P = np.array([[nodes[n]['z'], nodes[n]['y'], nodes[n]['x']] for n in c], float)
        U = P * S
        v = np.linalg.norm(np.diff(U, axis=0), axis=1) if len(c) > 1 else np.array([0.])
        e = np.array([ep.get(n, np.nan) for n in c[1:]]) if len(c) > 1 else np.array([np.nan])
        own = set(c); d_other = []; dens = []
        for n in c:
            ns, tr = trees[int(nodes[n]['t'])]
            dd, jj = tr.query(pos[n], k=min(3, len(ns)))
            dd = np.atleast_1d(dd); jj = np.atleast_1d(jj)
            oth = [d for d, j in zip(dd, jj) if ns[j] not in own]
            d_other.append(oth[0] if oth else 99.)
            dens.append(len(tr.query_ball_point(pos[n], 10.)) - 1)
        d_other = np.array(d_other)
        rows.append([len(c), nodes[c[0]]['t'], nodes[c[-1]]['t'], *P.mean(0), *P.std(0), float(v.mean()), float(v.max()),
                     float(np.nanmean(e)) if np.isfinite(e).any() else -1., float(np.nanmin(e)) if np.isfinite(e).any() else -1.,
                     (P[:, 0].mean() - zmin) / zptp, float(np.sqrt((P[:, 1].mean() - 128) ** 2 + (P[:, 2].mean() - 128) ** 2)), 0.,
                     float((d_other < 5).mean()), float((d_other < 7).mean()), float(np.median(d_other)), float(np.mean(dens)), npred, len(trk), 0.])
    X = np.array(rows, float)
    X[:, FEATS.index('lrank')] = np.argsort(np.argsort(lens)) / len(lens)
    X[:, FEATS.index('frac_iso')] = sum(lens) / npred
    return trk, X


def apply(nodes, edges, model_path, lam=1.0, tp_ref=600., ratio=1.04, max_len=None):
    import lgb_np
    trk, X = features(nodes, edges)
    if not trk: return nodes, edges, {'jp_removed_tracks': 0, 'jp_removed_nodes': 0}
    m = lgb_np.load(model_path)
    pr = np.exp(m.raw(X)) if str(getattr(m, 'objective', '')).startswith('poisson') else m.predict(X)
    npred = len(nodes); ntot_hat = npred * ratio; m_hat = 1 - 0.1 * (npred - ntot_hat) / ntot_hat
    be = 0.1 * tp_ref / ntot_hat / m_hat
    rm = set()
    for c, p, x in zip(trk, pr, X):
        if max_len is not None and len(c) > max_len: continue
        if p < lam * len(c) * be: rm.update(c)
    nodes = {n: v for n, v in nodes.items() if n not in rm}
    edges = [e for e in edges if int(e['source_id']) not in rm and int(e['target_id']) not in rm]
    return nodes, edges, {'jp_removed_tracks': int(sum(1 for c, p in zip(trk, pr) if c[0] in rm)), 'jp_removed_nodes': len(rm)}
