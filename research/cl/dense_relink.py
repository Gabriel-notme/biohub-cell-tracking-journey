"""Dense relinking: candidate links (s->d) supported by the base transformer's dense parent-assignment probabilities
(saved per movie as npz: edges [src_idx, tgt_idx, prob*1e6], coords [t,z,y,x]; node id == index in the candidate graph).
Accepting a candidate removes s's current single child link and d's current parent link, then adds s->d."""
from collections import defaultdict
from pathlib import Path
import numpy as np

S = np.array([1.625, .40625, .40625])
FEATS = ['typ', 'dp', 'dp_cur_d', 'dp_s_curd', 'dp_best_d', 'dp_rank_d', 'fe', 'dist', 'dz', 'd_cur_s', 'd_cur_d', 'hist_s', 'fut_d',
         'fut_curd', 'hist_curs', 'sp_s', 'dpred', 'dpred_cur', 'dpred_curs', 'cos_s', 'z', 'tt', 'dens_s', 'n_cand_d', 'n_cand_s']


def load_dense(path, nodes):
    """Return dict (s, d) -> prob restricted to node ids present in the graph whose candidate-graph coordinates match."""
    z = np.load(path); E = z['edges']; C = z['coords']
    ok = np.zeros(len(C), bool)
    for i in range(len(C)):  # node ids follow the candidate-graph index; centroid refinement may move nodes slightly
        v = nodes.get(i)
        if v is not None and int(v['t']) == int(C[i][0]) and np.linalg.norm((np.array([v['z'], v['y'], v['x']], float) - C[i][1:]) * S) <= 3.0:
            ok[i] = True
    dp = {}
    for a, b, p in E.tolist():
        if ok[a] and ok[b]: dp[(int(a), int(b))] = p / 1e6
    return dp


def features(nodes, edges, dp, fe):
    from scipy.spatial import cKDTree
    succ = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a
    pos = {n: np.array([v['z'], v['y'], v['x']], float) * S for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    trees = {t: cKDTree(np.array([pos[n] for n in ns])) for t, ns in frames.items()}
    T = max(frames)
    into = defaultdict(list)
    for (a, b), p in dp.items(): into[b].append(p)
    best = {b: max(v) for b, v in into.items()}
    ncs = defaultdict(int); ncd = defaultdict(int)

    def back(n, lim=30):
        k = 0; path = [n]
        while n in par and k < lim: n = par[n]; k += 1; path.append(n)
        return k, path

    def fwd(n, lim=30):
        k = 0
        while succ.get(n) and k < lim: n = succ[n][0]; k += 1
        return k
    rows = []
    for (s, d), p in dp.items():
        if p < 0.05 or s not in nodes or d not in nodes or d in succ.get(s, []): continue
        if int(nodes[d]['t']) != int(nodes[s]['t']) + 1: continue
        ks = succ.get(s, []); ps_ = par.get(d)
        if len(ks) >= 2: continue
        if ps_ is not None and len(succ.get(ps_, [])) >= 2: continue
        cur_d = ks[0] if ks else None
        if cur_d is not None and len(succ.get(cur_d, [])) >= 2: continue
        if par.get(s) is not None and len(succ.get(par[s], [])) >= 2 and ks: continue
        typ = (1 if cur_d is not None else 0) + (2 if ps_ is not None else 0)
        t = int(nodes[s]['t'])
        hs, path = back(s); vs = pos[path[0]] - pos[path[1]] if len(path) > 1 else np.zeros(3)
        disp = pos[d] - pos[s]; dist = float(np.linalg.norm(disp))
        hc, pc = back(ps_) if ps_ is not None else (-1, None)
        vc = (pos[pc[0]] - pos[pc[1]]) if (pc is not None and len(pc) > 1) else np.zeros(3)
        rk = sorted(into[d], reverse=True).index(p) if into.get(d) else -1
        rows.append(dict(s=s, d=d, cur_d=cur_d, cur_s=ps_, typ=typ, dp=p, dp_cur_d=dp.get((ps_, d), 0.) if ps_ is not None else -1.,
                         dp_s_curd=dp.get((s, cur_d), 0.) if cur_d is not None else -1., dp_best_d=best.get(d, 0.), dp_rank_d=rk,
                         fe=fe.get((s, d), -1.), dist=dist, dz=float(abs(disp[0])),
                         d_cur_s=float(np.linalg.norm(pos[cur_d] - pos[s])) if cur_d is not None else -1.,
                         d_cur_d=float(np.linalg.norm(pos[d] - pos[ps_])) if ps_ is not None else -1.,
                         hist_s=hs, fut_d=fwd(d), fut_curd=fwd(cur_d) if cur_d is not None else -1, hist_curs=hc,
                         sp_s=float(np.linalg.norm(vs)), dpred=float(np.linalg.norm(pos[d] - (pos[s] + vs))),
                         dpred_cur=float(np.linalg.norm(pos[cur_d] - (pos[s] + vs))) if cur_d is not None else -1.,
                         dpred_curs=float(np.linalg.norm(pos[d] - (pos[ps_] + vc))) if ps_ is not None else -1.,
                         cos_s=float(vs @ disp / (np.linalg.norm(vs) * dist + 1e-6)) if hs > 0 else 0.,
                         z=float(pos[s][0]), tt=t / T, dens_s=len(trees[t].query_ball_point(pos[s], 8.0))))
        ncs[s] += 1; ncd[d] += 1
    for r in rows: r['n_cand_s'] = ncs[r['s']]; r['n_cand_d'] = ncd[r['d']]
    return rows


def apply(nodes, edges, dense_path, fullgeff, model_path, th=0.5):
    import lgb_np, edge_link
    stats = {'dr_candidates': 0, 'dr_applied': 0}
    if not Path(dense_path).exists(): stats['dr_skipped'] = 'no_dense'; return nodes, edges, stats
    dp = load_dense(dense_path, nodes)
    fe = {}
    if fullgeff is not None and Path(fullgeff).exists():
        f = edge_link.load_full(fullgeff); fe = {(int(a), int(b)): float(p) for (a, b), p in zip(f[3].tolist(), f[4].tolist())}
    rows = features(nodes, edges, dp, fe)
    stats['dr_candidates'] = len(rows)
    if not rows: return nodes, edges, stats
    X = np.array([[r.get(k, -1) if r.get(k) is not None else -1 for k in FEATS] for r in rows], dtype=np.float32)
    p = lgb_np.load(model_path).predict(X)
    touched = set(); rm = set(); add = []
    for i in np.argsort(-p):
        if p[i] < th: break
        r = rows[i]; s, d, cd, cs = r['s'], r['d'], r['cur_d'], r['cur_s']
        if any(x in touched for x in (s, d, cd, cs) if x is not None): continue
        touched.update(x for x in (s, d, cd, cs) if x is not None)
        if cd is not None: rm.add((s, cd))
        if cs is not None: rm.add((cs, d))
        add.append((s, d, float(p[i])))
    ne = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in rm] + [{'source_id': s, 'target_id': d, 'dense_relink': round(q, 4)} for s, d, q in add]
    stats['dr_applied'] = len(add)
    return nodes, ne, stats
