"""Extended relinking (post-hoc on final graphs). Two candidate types, not limited to the pre-ILP candidate graph:
  IN : s is a track end at t (no child); d at t+1 within R um already has a parent y (y has only d as child). Action: y->d  =>  s->d.
  OUT: d is a track start at t+1 (no parent); s at t within R um has exactly one child x.            Action: s->x  =>  s->d.
Forks, fork daughters and fork parents are never touched. Features are purely geometric / structural (embryo-agnostic)."""
from collections import defaultdict
import numpy as np

S = np.array([1.625, .40625, .40625])
R = 10.0
FEATS = ['typ', 'd_new', 'd_old', 'd_ratio', 'dz_new', 'dz_old', 'pred_new', 'pred_old', 'pred_new_o', 'pred_old_o', 'hist_s', 'fut_d', 'hist_o', 'fut_o',
         'fe_new', 'fe_old', 'ep_old', 'rank_new', 'n_near', 'dens', 'tt', 'z', 'sp_s', 'sp_o', 'cos_new', 'cos_old']


def _f(v):
    try:
        return float(v) if v is not None else -1.
    except (TypeError, ValueError):
        return -1.


def features(nodes, edges, full=None):
    from scipy.spatial import cKDTree
    succ = defaultdict(list); par = {}; eattr = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a; eattr[(a, b)] = e
    pos = {n: np.array([v['z'], v['y'], v['x']], float) * S for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    trees = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in frames.items()}
    fedge = {}
    if full is not None:
        fids, fT, fV, fE, fprob = full
        fedge = {(int(a), int(b)): float(p) for (a, b), p in zip(fE.tolist(), fprob.tolist())}
    T = max(frames)
    infork = set()
    for a, ch in succ.items():
        if len(ch) >= 2: infork.add(a); infork.update(ch)

    def back(n, lim=30):
        k = 0; path = [n]
        while n in par and k < lim: n = par[n]; k += 1; path.append(n)
        return k, path

    def fwd(n, lim=30):
        k = 0; path = [n]
        while len(succ.get(n, [])) >= 1 and k < lim: n = succ[n][0]; k += 1; path.append(n)
        return k, path

    def vel_back(n):
        k, p = back(n, 2)
        return (pos[p[0]] - pos[p[1]]) if len(p) > 1 else None

    def vel_fwd(n):
        k, p = fwd(n, 2)
        return (pos[p[1]] - pos[p[0]]) if len(p) > 1 else None

    def pred_err(src, dst):  # motion-predicted position of src at t+1 vs dst; and dst's back-extrapolation vs src
        vs = vel_back(src); vd = vel_fwd(dst)
        e1 = float(np.linalg.norm(pos[src] + vs - pos[dst])) if vs is not None else -1.
        e2 = float(np.linalg.norm(pos[dst] - vd - pos[src])) if vd is not None else -1.
        return e1, e2

    def cosang(src, dst):
        vs = vel_back(src)
        if vs is None: return 0.
        d = pos[dst] - pos[src]
        return float(vs @ d / (np.linalg.norm(vs) * np.linalg.norm(d) + 1e-6))
    rows = []
    for s in nodes:
        t = int(nodes[s]['t'])
        if t + 1 not in trees or s in infork: continue
        ns, tr = trees[t + 1]
        near = tr.query_ball_point(pos[s], R)
        cand = sorted(((float(np.linalg.norm(pos[ns[j]] - pos[s])), ns[j]) for j in near))
        ks = succ.get(s, [])
        for rank, (dist, d) in enumerate(cand):
            if d in infork: continue
            if not ks and d in par:  # IN
                y = par[d]
                if len(succ[y]) != 1 or y in infork: continue
                typ, o, old = 0, y, (y, d)
            elif len(ks) == 1 and d not in par and ks[0] != d:  # OUT
                x = ks[0]
                if x in infork: continue
                typ, o, old = 1, x, (s, x)
            else:
                continue
            d_old = float(np.linalg.norm(pos[old[1]] - pos[old[0]]))
            pn1, pn2 = pred_err(s, d); po1, po2 = pred_err(*old)
            hs = back(s)[0]; fd = fwd(d)[0]
            if typ == 0: ho, fo = back(o)[0], -1
            else: ho, fo = -1, fwd(o)[0]
            vo = vel_back(old[0])
            r = dict(s=s, d=d, o=o, typ=typ, t=t, d_new=dist, d_old=d_old, d_ratio=dist / (d_old + 1e-3),
                     dz_new=float(abs(pos[d][0] - pos[s][0])), dz_old=float(abs(pos[old[1]][0] - pos[old[0]][0])),
                     pred_new=pn1, pred_new_o=pn2, pred_old=po1, pred_old_o=po2, hist_s=hs, fut_d=fd, hist_o=ho, fut_o=fo,
                     fe_new=fedge.get((s, d), -1.), fe_old=fedge.get(old, -1.), ep_old=_f(eattr[old].get('edge_prob')),
                     rank_new=rank, n_near=len(cand), dens=len(trees[t][1].query_ball_point(pos[s], 8.0)), tt=t / T, z=float(pos[s][0]),
                     sp_s=float(np.linalg.norm(vel_back(s))) if vel_back(s) is not None else -1., sp_o=float(np.linalg.norm(vo)) if vo is not None else -1.,
                     cos_new=cosang(s, d), cos_old=cosang(*old))
            rows.append(r)
    return rows


def apply_rows(nodes, edges, rows, p, th):
    order = np.argsort(-p)
    touched = set(); rm = set(); add = []
    for i in order:
        if p[i] < th: break
        r = rows[i]; s, d, o = r['s'], r['d'], r['o']
        if s in touched or d in touched or o in touched: continue
        touched.update((s, d, o))
        rm.add((o, d) if r['typ'] == 0 else (s, o)); add.append((s, d, float(p[i])))
    ne = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in rm] + [{'source_id': s, 'target_id': d, 'xrl': round(q, 4)} for s, d, q in add]
    return nodes, ne, dict(xrl_applied=len(add))


def apply(nodes, edges, full, model_path, th=0.5):
    import lgb_np
    rows = features(nodes, edges, full)
    if not rows: return nodes, edges, dict(xrl_applied=0)
    X = np.array([[r.get(k, -1) for k in FEATS] for r in rows], dtype=np.float32)
    return apply_rows(nodes, edges, rows, lgb_np.load(model_path).predict(X), th)
