"""Learned relinking: candidate links (s->d) present in the pre-ILP candidate graph but absent from the final graph,
where s and/or d are currently linked elsewhere. Accepting a candidate removes s's current single child link and
d's current parent link, then adds s->d."""
from collections import defaultdict
from pathlib import Path
import numpy as np

S = np.array([1.625, .40625, .40625])
FEATS = ['typ', 'fe', 'dist', 'dz', 'd_cur_s', 'd_cur_d', 'fe_cur_s', 'fe_cur_d', 'ep_cur_s', 'ep_cur_d', 'hist_s', 'fut_d', 'fut_curd', 'hist_curs',
         'sp_s', 'dpred', 'dpred_cur', 'cos_s', 'cos_cur', 'z', 'tt', 'dens_s', 'nkids_s', 'orphan_fut', 'widow_hist']


def _f(v):
    try:
        return float(v) if v is not None else -1.
    except (TypeError, ValueError):
        return -1.


def features(nodes, edges, full):
    from scipy.spatial import cKDTree
    succ = defaultdict(list); par = {}; eattr = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a; eattr[(a, b)] = e
    pos = {n: np.array([v['z'], v['y'], v['x']], float) * S for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    trees = {t: cKDTree(np.array([pos[n] for n in ns])) for t, ns in frames.items()}
    fids, fT, fV, fE, fprob = full
    fedge = {(int(a), int(b)): float(p) for (a, b), p in zip(fE.tolist(), fprob.tolist())}
    T = max(frames)

    def back(n, lim=30):
        k = 0; path = [n]
        while n in par and k < lim: n = par[n]; k += 1; path.append(n)
        return k, path

    def fwd(n, lim=30):
        k = 0
        while len(succ.get(n, [])) >= 1 and k < lim: n = succ[n][0]; k += 1
        return k
    rows = []
    for (s, d), p in fedge.items():
        if s not in nodes or d not in nodes or (s, d) in eattr: continue
        if int(nodes[d]['t']) != int(nodes[s]['t']) + 1: continue
        ks = succ.get(s, []); ps_ = par.get(d)
        if len(ks) == 0 and ps_ is None: continue  # pure free-end case handled by edge_link
        if len(ks) >= 2: continue  # do not touch forks
        if ps_ is not None and len(succ.get(ps_, [])) >= 2: continue  # d is a fork daughter: keep the division
        if par.get(s) is not None and len(succ.get(par[s], [])) >= 2 and ks: continue  # s is a fresh daughter: keep its branch
        cur_d = ks[0] if ks else None
        if cur_d is not None and len(succ.get(cur_d, [])) >= 2: continue  # current child divides next: keep
        typ = (1 if cur_d is not None else 0) + (2 if ps_ is not None else 0)
        t = int(nodes[s]['t'])
        hs, path = back(s)
        vs = pos[path[0]] - pos[path[1]] if len(path) > 1 else np.zeros(3)
        disp = pos[d] - pos[s]; dist = float(np.linalg.norm(disp))
        r = dict(s=s, d=d, cur_d=cur_d, cur_s=ps_, t=t, typ=typ, fe=p, dist=dist, dz=float(abs(disp[0])),
                 d_cur_s=float(np.linalg.norm(pos[cur_d] - pos[s])) if cur_d is not None else -1.,
                 d_cur_d=float(np.linalg.norm(pos[d] - pos[ps_])) if ps_ is not None else -1.,
                 fe_cur_s=fedge.get((s, cur_d), -1.) if cur_d is not None else -2.,
                 fe_cur_d=fedge.get((ps_, d), -1.) if ps_ is not None else -2.,
                 ep_cur_s=_f(eattr[(s, cur_d)].get('edge_prob')) if cur_d is not None else -2.,
                 ep_cur_d=_f(eattr[(ps_, d)].get('edge_prob')) if ps_ is not None else -2.,
                 hist_s=hs, fut_d=fwd(d), fut_curd=fwd(cur_d) if cur_d is not None else -1,
                 hist_curs=back(ps_)[0] if ps_ is not None else -1,
                 sp_s=float(np.linalg.norm(vs)), dpred=float(np.linalg.norm(pos[d] - (pos[s] + vs))),
                 dpred_cur=float(np.linalg.norm(pos[cur_d] - (pos[s] + vs))) if cur_d is not None else -1.,
                 cos_s=float(vs @ disp / (np.linalg.norm(vs) * dist + 1e-6)) if hs > 0 else 0.,
                 cos_cur=float(vs @ (pos[cur_d] - pos[s]) / (np.linalg.norm(vs) * np.linalg.norm(pos[cur_d] - pos[s]) + 1e-6)) if (hs > 0 and cur_d is not None) else 0.,
                 z=float(pos[s][0]), tt=t / T, dens_s=len(trees[t].query_ball_point(pos[s], 8.0)), nkids_s=len(ks),
                 orphan_fut=fwd(cur_d) if cur_d is not None else -1, widow_hist=back(ps_)[0] if ps_ is not None else -1)
        rows.append(r)
    return rows


def apply(nodes, edges, full, model_path, th=0.5):
    import lgb_np
    rows = features(nodes, edges, full)
    stats = {'rl_candidates': len(rows), 'rl_applied': 0}
    if not rows: return nodes, edges, stats
    bst = lgb_np.load(model_path)
    X = np.array([[r.get(k, -1) for k in FEATS] for r in rows], dtype=np.float32)
    p = bst.predict(X)
    order = np.argsort(-p)
    touched = set(); rm = set(); add = []
    for i in order:
        if p[i] < th: break
        r = rows[i]; s, d, cd, cs = r['s'], r['d'], r['cur_d'], r['cur_s']
        if s in touched or d in touched or (cd is not None and cd in touched) or (cs is not None and cs in touched): continue
        touched.update(x for x in (s, d, cd, cs) if x is not None)
        if cd is not None: rm.add((s, cd))
        if cs is not None: rm.add((cs, d))
        add.append((s, d, float(p[i])))
    ne = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in rm] + [{'source_id': s, 'target_id': d, 'relink': round(q, 4)} for s, d, q in add]
    stats['rl_applied'] = len(add)
    return nodes, ne, stats
