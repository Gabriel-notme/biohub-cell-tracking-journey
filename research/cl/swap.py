"""Learned swap repair: for two nearby links s->d1 and s2->d2 in the same frame, consider the crossed assignment
s->d2, s2->d1. Features compare both assignments (distance, motion prediction, pre-ILP candidate edges, track lengths)."""
from collections import defaultdict
import numpy as np

S = np.array([1.625, .40625, .40625])
FEATS = ['c_cur', 'c_alt', 'c_diff', 'm_cur', 'm_alt', 'm_diff', 'fe_a1', 'fe_a2', 'fe_c1', 'fe_c2', 'fe_alt_n', 'fe_cur_n', 'd_ss', 'd_dd',
         'h_s', 'h_s2', 'f_d1', 'f_d2', 'sp_s', 'sp_s2', 'z', 'tt', 'dens', 'ep_c1', 'ep_c2', 'cos_cur', 'cos_alt']


def _f(v):
    try:
        return float(v) if v is not None else -1.
    except (TypeError, ValueError):
        return -1.


def features(nodes, edges, full, R=9.0):
    from scipy.spatial import cKDTree
    succ = defaultdict(list); par = {}; eattr = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a; eattr[(a, b)] = e
    pos = {n: np.array([v['z'], v['y'], v['x']], float) * S for n, v in nodes.items()}
    fedge = {}
    if full is not None:
        fids, fT, fV, fE, fprob = full
        fedge = {(int(a), int(b)): float(p) for (a, b), p in zip(fE.tolist(), fprob.tolist())}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
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
    for t, ns in frames.items():
        src = [n for n in ns if len(succ.get(n, [])) == 1 and len(succ.get(par.get(n, -1), [])) < 2]
        src = [n for n in src if len(succ.get(succ[n][0], [])) < 2]
        if len(src) < 2: continue
        tr = cKDTree(np.array([pos[n] for n in src]))
        for i, j in tr.query_pairs(R):
            s, s2 = src[i], src[j]
            d1, d2 = succ[s][0], succ[s2][0]
            hs, ps = back(s); hs2, ps2 = back(s2)
            vs = pos[ps[0]] - pos[ps[1]] if len(ps) > 1 else np.zeros(3)
            vs2 = pos[ps2[0]] - pos[ps2[1]] if len(ps2) > 1 else np.zeros(3)
            c_cur = np.linalg.norm(pos[d1] - pos[s]) + np.linalg.norm(pos[d2] - pos[s2])
            c_alt = np.linalg.norm(pos[d2] - pos[s]) + np.linalg.norm(pos[d1] - pos[s2])
            m_cur = np.linalg.norm(pos[d1] - pos[s] - vs) + np.linalg.norm(pos[d2] - pos[s2] - vs2)
            m_alt = np.linalg.norm(pos[d2] - pos[s] - vs) + np.linalg.norm(pos[d1] - pos[s2] - vs2)
            if c_alt - c_cur > 12.0 and m_alt - m_cur > 12.0: continue
            fa1, fa2 = fedge.get((s, d2), -1.), fedge.get((s2, d1), -1.)
            fc1, fc2 = fedge.get((s, d1), -1.), fedge.get((s2, d2), -1.)

            def cosv(v, w):
                return float(v @ w / (np.linalg.norm(v) * np.linalg.norm(w) + 1e-6)) if np.linalg.norm(v) > 0 else 0.
            r = dict(s=s, s2=s2, d1=d1, d2=d2, t=t, c_cur=float(c_cur), c_alt=float(c_alt), c_diff=float(c_alt - c_cur), m_cur=float(m_cur), m_alt=float(m_alt),
                     m_diff=float(m_alt - m_cur), fe_a1=fa1, fe_a2=fa2, fe_c1=fc1, fe_c2=fc2, fe_alt_n=int(fa1 >= 0) + int(fa2 >= 0), fe_cur_n=int(fc1 >= 0) + int(fc2 >= 0),
                     d_ss=float(np.linalg.norm(pos[s] - pos[s2])), d_dd=float(np.linalg.norm(pos[d1] - pos[d2])), h_s=hs, h_s2=hs2, f_d1=fwd(d1), f_d2=fwd(d2),
                     sp_s=float(np.linalg.norm(vs)), sp_s2=float(np.linalg.norm(vs2)), z=float(pos[s][0]), tt=t / T,
                     dens=len(tr.query_ball_point(pos[s], 8.0)), ep_c1=_f(eattr[(s, d1)].get('edge_prob')), ep_c2=_f(eattr[(s2, d2)].get('edge_prob')),
                     cos_cur=(cosv(vs, pos[d1] - pos[s]) + cosv(vs2, pos[d2] - pos[s2])) / 2, cos_alt=(cosv(vs, pos[d2] - pos[s]) + cosv(vs2, pos[d1] - pos[s2])) / 2)
            rows.append(r)
    return rows


def apply(nodes, edges, full, model_path, th=0.6):
    import lgb_np
    rows = features(nodes, edges, full)
    stats = {'sw_candidates': len(rows), 'sw_applied': 0}
    if not rows: return nodes, edges, stats
    bst = lgb_np.load(model_path)
    X = np.array([[r.get(k, -1) for k in FEATS] for r in rows], dtype=np.float32)
    p = bst.predict(X)
    order = np.argsort(-p)
    used = set(); rm = set(); add = []
    for i in order:
        if p[i] < th: break
        r = rows[i]
        ks = (r['s'], r['s2'], r['d1'], r['d2'])
        if any(k in used for k in ks): continue
        used.update(ks)
        rm.add((r['s'], r['d1'])); rm.add((r['s2'], r['d2'])); add += [(r['s'], r['d2'], float(p[i])), (r['s2'], r['d1'], float(p[i]))]
    ne = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in rm] + [{'source_id': a, 'target_id': b, 'swap': round(q, 4)} for a, b, q in add]
    stats['sw_applied'] = len(add) // 2
    return nodes, ne, stats
