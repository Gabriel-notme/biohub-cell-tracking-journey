"""Learned division completion: score start/stolen fork candidates (from div_complete.candidates + b1 fork/edge
heads) with a small LightGBM model over b1 probabilities and track-structure features, then accept greedily."""
from collections import defaultdict
import numpy as np

S = np.array([1.625, .40625, .40625])
FEATS = ['is_stolen', 'fork', 'e_old', 'e_pb', 'd_pb', 'd_ab', 'd_pa', 't', 'z', 'hist_p', 'fut_a', 'fut_b', 'cos_ab', 'vel_p', 'dens_p', 'dens_b',
         'ncand_p', 'ncand_b', 'hist_q', 'd_qb', 'd_qp', 'qstart_dp', 'qstart_dt', 'fork_rank_p', 'fork_rank_b', 'fork_gap_p']


def add_features(nodes, edges, cands):
    """cands: list of dicts from div_complete.score_candidates(..., with_edges=True). Adds structure features in place."""
    from scipy.spatial import cKDTree
    succ = defaultdict(list); par = {}
    for e in edges:
        a_, b_ = int(e['source_id']), int(e['target_id']); succ[a_].append(b_); par[b_] = a_
    pos = {n: np.array([v['z'], v['y'], v['x']], float) * S for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    trees = {t: cKDTree(np.array([pos[n] for n in ns])) for t, ns in frames.items()}

    def fwd(n, lim=40):
        k = 0
        while len(succ.get(n, [])) == 1 and k < lim: n = succ[n][0]; k += 1
        return k

    def back(n, lim=40):
        k = 0
        while n in par and k < lim: n = par[n]; k += 1
        return k, n
    ncp = defaultdict(int); ncb = defaultdict(int); byp = defaultdict(list); byb = defaultdict(list)
    for c in cands: ncp[c['p']] += 1; ncb[c['b']] += 1; byp[c['p']].append(c['fork']); byb[c['b']].append(c['fork'])
    for c in cands:
        p, a, b, q = c['p'], c['a'], c['b'], c['q']
        t = int(nodes[p]['t']); hp, _ = back(p); pp = par.get(p)
        vp = pos[p] - pos[pp] if pp is not None else np.zeros(3)
        va, vb = pos[a] - pos[p], pos[b] - pos[p]
        c.update(t=t, z=float(pos[p][0]), hist_p=hp, fut_a=fwd(a), fut_b=fwd(b),
                 cos_ab=float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-6)), d_pa=float(np.linalg.norm(va)), vel_p=float(np.linalg.norm(vp)),
                 dens_p=len(trees[t].query_ball_point(pos[p], 8.0)), dens_b=len(trees[t + 1].query_ball_point(pos[b], 8.0)), ncand_p=ncp[p], ncand_b=ncb[b],
                 is_stolen=int(c['typ'] == 'stolen'))
        if q is not None:
            hq, qs = back(q)
            c.update(hist_q=hq, d_qb=float(np.linalg.norm(pos[b] - pos[q])), d_qp=float(np.linalg.norm(pos[q] - pos[p])),
                     qstart_dp=float(np.linalg.norm(pos[qs] - pos[p])), qstart_dt=t - int(nodes[qs]['t']))
        sp = sorted(byp[p], reverse=True); sb = sorted(byb[b], reverse=True)
        c['fork_rank_p'] = sp.index(c['fork']); c['fork_rank_b'] = sb.index(c['fork'])
        c['fork_gap_p'] = c['fork'] - (sp[1] if len(sp) > 1 and sp[0] == c['fork'] else sp[0])
    return cands


def predict(cands, model_path):
    import lightgbm as lgb
    if not cands: return np.zeros(0)
    X = np.array([[(-1 if c.get(k) is None else c[k]) for k in FEATS] for c in cands], dtype=np.float32)
    return lgb.Booster(model_file=str(model_path)).predict(X)


def apply(nodes, edges, cands, prob, th=0.5):
    succ = defaultdict(list)
    for e in edges: succ[int(e['source_id'])].append(int(e['target_id']))
    order = np.argsort(-np.asarray(prob))
    used_p = set(); used_b = set(); rm = set(); add = []
    for i in order:
        if prob[i] < th: break
        c = cands[i]; p, a, b, q = c['p'], c['a'], c['b'], c['q']
        if p in used_p or b in used_b or a in used_b: continue
        if c['typ'] == 'stolen':
            if q in used_p or q in used_b: continue
            rm.add((q, b)); used_p.add(q)
        used_p.add(p); used_b.add(b); used_b.add(a); add.append((p, b, float(prob[i])))
    ne = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in rm] + [{'source_id': p, 'target_id': b, 'div_complete': 1, 'div_learn': round(q, 4)} for p, b, q in add]
    return ne, {'div_added': len(add), 'div_stolen': len(rm)}
