"""Learned fragment gap filling: re-insert chains of pre-ILP detections that the ILP dropped when they continue a
current track end (before side) and/or lead into a current track start (after side), scored by a small GBM."""
from collections import defaultdict
from pathlib import Path
import numpy as np

S = np.array([1.625, .40625, .40625])
R_ATTACH = 10.0
FEATS = ['L', 'side_b', 'side_a', 'de', 'ds', 'fe_b', 'fe_a', 'pin_mean', 'pin_min', 'dup_min', 'dup_mean', 'nn_mean', 'hist_e', 'fut_s',
         'dpred_e', 'cos_e', 'step_mean', 'step_max', 'z', 'tt', 'dens', 'n_comp_e', 'n_comp_s', 'rk_e', 'rk_s']


def chains(nodes, full):
    fids, fT, fV, fE, fprob = full
    idx = {int(i): j for j, i in enumerate(fids.tolist())}
    drop = np.array([int(i) not in nodes for i in fids.tolist()])
    fs = {}; fpar = {}; fp = {}
    for (a, b), p in zip(fE.tolist(), fprob.tolist()):
        fp[(a, b)] = p
        if drop[idx[a]] and drop[idx[b]]: fs[a] = b; fpar[b] = a
    out = []
    for j in np.where(drop)[0]:
        n0 = int(fids[j])
        if n0 in fpar: continue
        ch = [n0]
        while ch[-1] in fs and len(ch) < 200: ch.append(fs[ch[-1]])
        out.append(ch)
    return out, idx, fp


def features(nodes, edges, full):
    from scipy.spatial import cKDTree
    fids, fT, fV, fE, fprob = full
    fP = fV * S
    succ = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a
    pos = {n: np.array([v['z'], v['y'], v['x']], float) * S for n, v in nodes.items()}
    ends = defaultdict(list); starts = defaultdict(list); allf = defaultdict(list)
    for n, v in nodes.items():
        t = int(v['t']); allf[t].append(n)
        if not succ.get(n): ends[t].append(n)
        if n not in par: starts[t].append(n)
    etree = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in ends.items() if ns}
    stree = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in starts.items() if ns}
    atree = {t: cKDTree(np.array([pos[n] for n in ns])) for t, ns in allf.items()}
    T = max(allf)
    chs, idx, fp = chains(nodes, full)

    def back(n, lim=30):
        k = 0; path = [n]
        while n in par and k < lim: n = par[n]; k += 1; path.append(n)
        return k, path

    def fwd(n, lim=30):
        k = 0
        while succ.get(n) and k < lim: n = succ[n][0]; k += 1
        return k
    rows = []
    for ch in chs:
        ti = [int(fT[idx[x]]) for x in ch]; P = np.array([fP[idx[x]] for x in ch])
        t0, t1 = ti[0], ti[-1]
        e_c = etree[t0 - 1][0] if (t0 - 1) in etree else []
        s_c = stree[t1 + 1][0] if (t1 + 1) in stree else []
        de, e_n, ds, s_n = 99., None, 99., None
        if e_c:
            d, j = etree[t0 - 1][1].query(P[0]); de, e_n = float(d), e_c[int(j)]
        if s_c:
            d, j = stree[t1 + 1][1].query(P[-1]); ds, s_n = float(d), s_c[int(j)]
        side_b, side_a = int(de <= R_ATTACH), int(ds <= R_ATTACH)
        if not (side_b or side_a): continue
        dups = [atree[t].query(p)[0] if t in atree else 99. for t, p in zip(ti, P)]
        nn = [float(atree[t].query(p, k=2)[0][-1]) if t in atree and len(allf[t]) > 1 else 99. for t, p in zip(ti, P)]
        if min(dups) < 2.5: continue  # duplicates of existing detections
        pin = [fp.get((a, b), 0.) for a, b in zip(ch[:-1], ch[1:])]
        steps = np.linalg.norm(np.diff(P, axis=0), axis=1) if len(ch) > 1 else np.array([0.])
        he, dpred_e, cos_e = -1, 99., 0.
        if side_b:
            he, path = back(e_n)
            v = pos[path[0]] - pos[path[1]] if len(path) > 1 else np.zeros(3)
            dpred_e = float(np.linalg.norm(P[0] - (pos[e_n] + v)))
            disp = P[0] - pos[e_n]
            cos_e = float(v @ disp / (np.linalg.norm(v) * np.linalg.norm(disp) + 1e-6)) if he > 0 else 0.
        r = dict(chain=ch, e=e_n if side_b else None, s=s_n if side_a else None, t0=t0, t1=t1, L=len(ch), side_b=side_b, side_a=side_a,
                 de=de, ds=ds, fe_b=fp.get((e_n, ch[0]), -1.) if side_b else -2., fe_a=fp.get((ch[-1], s_n), -1.) if side_a else -2.,
                 pin_mean=float(np.mean(pin)) if pin else -1., pin_min=float(np.min(pin)) if pin else -1.,
                 dup_min=float(min(dups)), dup_mean=float(np.mean(dups)), nn_mean=float(np.mean(nn)), hist_e=he, fut_s=fwd(s_n) if side_a else -1,
                 dpred_e=dpred_e, cos_e=cos_e, step_mean=float(steps.mean()), step_max=float(steps.max()), z=float(P[0][0]), tt=t0 / T,
                 dens=len(atree[t0].query_ball_point(P[0], 8.0)) if t0 in atree else 0)
        rows.append(r)
    ce = defaultdict(list); cs = defaultdict(list)
    for r in rows:
        if r['e'] is not None: ce[r['e']].append(r['de'])
        if r['s'] is not None: cs[r['s']].append(r['ds'])
    for r in rows:
        r['n_comp_e'] = len(ce[r['e']]) if r['e'] is not None else 0; r['n_comp_s'] = len(cs[r['s']]) if r['s'] is not None else 0
        r['rk_e'] = sorted(ce[r['e']]).index(r['de']) if r['e'] is not None else -1
        r['rk_s'] = sorted(cs[r['s']]).index(r['ds']) if r['s'] is not None else -1
    return rows


def apply(nodes, edges, fullgeff, model_path, th=0.5, full=None):
    import lgb_np
    if full is None:
        import edge_link
        full = edge_link.load_full(fullgeff)
    fids, fT, fV, fE, fprob = full
    idx = {int(i): j for j, i in enumerate(fids.tolist())}
    rows = features(nodes, edges, full)
    stats = {'fl_candidates': len(rows), 'fl_chains': 0, 'fl_nodes': 0}
    if not rows: return nodes, edges, stats
    X = np.array([[r.get(k, -1) for k in FEATS] for r in rows], dtype=np.float32)
    p = lgb_np.load(model_path).predict(X)
    succ = defaultdict(list); par = {}
    for e in edges: succ[int(e['source_id'])].append(int(e['target_id'])); par[int(e['target_id'])] = int(e['source_id'])
    nn = dict(nodes); ne = list(edges); nid = max(int(k) for k in nodes) + 1
    used_e, used_s, used_node = set(), set(), set()
    for i in np.argsort(-p):
        if p[i] < th: break
        r = rows[i]
        if any(x in used_node for x in r['chain']): continue
        e_n = r['e'] if (r['e'] is not None and r['e'] not in used_e and not succ.get(r['e'])) else None
        s_n = r['s'] if (r['s'] is not None and r['s'] not in used_s and r['s'] not in par) else None
        if e_n is None and s_n is None: continue
        used_node.update(r['chain'])
        new = []
        for x in r['chain']:
            j = idx[x]
            nn[nid] = {'node_id': nid, 't': int(fT[j]), 'z': float(fV[j][0]), 'y': float(fV[j][1]), 'x': float(fV[j][2]), 'frag_link': 1}
            new.append(nid); nid += 1
        for a, b in zip(new[:-1], new[1:]): ne.append({'source_id': a, 'target_id': b, 'frag_link': 1})
        if e_n is not None: ne.append({'source_id': e_n, 'target_id': new[0], 'frag_link': 1}); used_e.add(e_n); succ[e_n].append(new[0])
        if s_n is not None: ne.append({'source_id': new[-1], 'target_id': s_n, 'frag_link': 1}); used_s.add(s_n); par[s_n] = new[-1]
        stats['fl_chains'] += 1; stats['fl_nodes'] += len(new)
    return nn, ne, stats
