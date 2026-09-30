"""Learned free-end linking: connect a track end at t to a track start at t+1 (or t+2 via one inserted node)
when a small LightGBM model, driven mainly by the pre-ILP candidate-graph edge probability, is confident."""
from collections import defaultdict
from pathlib import Path
import numpy as np

S = np.array([1.625, .40625, .40625])
RMAX = {1: 14.0, 2: 18.0}
FEATS = ['gap', 'dist', 'dz', 'dxy', 'hist_s', 'fut_d', 'sp_s', 'sp_d', 'dpred', 'dpred_d', 'cos_s', 'cos_d', 'z', 'tt', 'dens_s', 'dens_d',
         'nn_d_prev', 'nn_s_next', 'fe', 'drop_mid', 'n_s', 'n_d', 'rk_s', 'rk_d', 'gap_s', 'gap_d']


def load_full(fullgeff):
    import zarr
    fg = zarr.open_group(str(fullgeff), mode='r')
    fids = np.asarray(fg['nodes/ids'][:]).astype(np.int64); fT = np.asarray(fg['nodes/props/t/values'][:]).astype(int)
    fV = np.stack([np.asarray(fg['nodes/props/%s/values' % k][:]) for k in 'zyx'], 1)
    fE = np.asarray(fg['edges/ids'][:]).astype(np.int64); fprob = np.asarray(fg['edges/props/edge_prob/values'][:])
    return fids, fT, fV, fE, fprob


def features(nodes, edges, full):
    from scipy.spatial import cKDTree
    succ = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a
    pos = {n: np.array([v['z'], v['y'], v['x']], float) * S for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    trees = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in frames.items()}
    fedge = {}; dtrees = {}
    if full is not None:
        fids, fT, fV, fE, fprob = full
        fP = fV * S
        fedge = {(int(a), int(b)): float(p) for (a, b), p in zip(fE.tolist(), fprob.tolist())}
        drop_idx = defaultdict(list)
        for i, (fi, t) in enumerate(zip(fids.tolist(), fT.tolist())):
            if fi not in nodes: drop_idx[t].append(i)
        dtrees = {t: (ix, cKDTree(fP[ix]), fP) for t, ix in drop_idx.items() if ix}
    ends = [n for n in nodes if not succ.get(n)]
    starts = defaultdict(list)
    for n in nodes:
        if n not in par: starts[int(nodes[n]['t'])].append(n)
    stree = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in starts.items() if ns}

    def back(n, lim=30):
        k = 0; path = [n]
        while n in par and k < lim: n = par[n]; k += 1; path.append(n)
        return k, path

    def fwd(n, lim=30):
        k = 0; path = [n]
        while len(succ.get(n, [])) >= 1 and k < lim: n = succ[n][0]; k += 1; path.append(n)
        return k, path
    T = max(frames)
    rows = []
    for sn in ends:
        t = int(nodes[sn]['t'])
        hs, ps = back(sn)
        vs = pos[ps[0]] - pos[ps[1]] if len(ps) > 1 else np.zeros(3)
        for gap in (1, 2):
            if t + gap not in stree: continue
            ns, tr = stree[t + gap]
            for j in tr.query_ball_point(pos[sn], RMAX[gap]):
                dn = ns[j]
                fd, pd_ = fwd(dn)
                vd = pos[pd_[1]] - pos[pd_[0]] if len(pd_) > 1 else np.zeros(3)
                disp = pos[dn] - pos[sn]
                pred_s = pos[sn] + gap * vs
                dist = float(np.linalg.norm(disp))
                r = dict(s=sn, d=dn, t=t, gap=gap, dist=dist, dz=float(abs(disp[0])), dxy=float(np.linalg.norm(disp[1:])),
                         hist_s=hs, fut_d=fd, sp_s=float(np.linalg.norm(vs)), sp_d=float(np.linalg.norm(vd)),
                         dpred=float(np.linalg.norm(pos[dn] - pred_s)), dpred_d=float(np.linalg.norm(pos[sn] - (pos[dn] - gap * vd))),
                         cos_s=float(vs @ disp / (np.linalg.norm(vs) * dist + 1e-6)) if hs > 0 else 0.0,
                         cos_d=float(vd @ disp / (np.linalg.norm(vd) * dist + 1e-6)) if fd > 0 else 0.0,
                         z=float(pos[sn][0]), tt=t / T,
                         dens_s=len(trees[t][1].query_ball_point(pos[sn], 8.0)), dens_d=len(trees[t + gap][1].query_ball_point(pos[dn], 8.0)),
                         nn_d_prev=float(trees[t + gap - 1][1].query(pos[dn], k=1)[0]) if (t + gap - 1) in trees else 99.,
                         nn_s_next=float(trees[t + 1][1].query(pos[sn], k=1)[0]) if (t + 1) in trees else 99.,
                         fe=fedge.get((sn, dn), -1.0))
                if (t + 1) in dtrees:
                    ix, dtr, fP = dtrees[t + 1]; mid = pos[sn] + disp / gap
                    dd, jj = dtr.query(mid, k=1); r['drop_mid'] = float(dd)
                    r['drop_xyz'] = [float(v) for v in fP[ix[int(jj)]] / S]
                else:
                    r['drop_mid'] = 99.
                rows.append(r)
    cs = defaultdict(list); cd = defaultdict(list)
    for r in rows: cs[(r['s'], r['gap'])].append(r['dist']); cd[(r['d'], r['gap'])].append(r['dist'])
    for r in rows:
        a = sorted(cs[(r['s'], r['gap'])]); b = sorted(cd[(r['d'], r['gap'])])
        r['n_s'] = len(a); r['n_d'] = len(b); r['rk_s'] = a.index(r['dist']); r['rk_d'] = b.index(r['dist'])
        r['gap_s'] = (a[1] - r['dist']) if (r['rk_s'] == 0 and len(a) > 1) else (r['dist'] - a[0])
        r['gap_d'] = (b[1] - r['dist']) if (r['rk_d'] == 0 and len(b) > 1) else (r['dist'] - b[0])
    return rows


def apply(nodes, edges, fullgeff, model_path, th=0.5, allow_gap2=True, snap_um=3.0):
    import lgb_np
    full = load_full(fullgeff) if fullgeff is not None and Path(fullgeff).exists() else None
    rows = features(nodes, edges, full)
    stats = {'el_candidates': len(rows), 'el_gap1': 0, 'el_gap2': 0}
    if not rows: return nodes, edges, stats
    bst = lgb_np.load(model_path)
    X = np.array([[r.get(k, -1) for k in FEATS] for r in rows], dtype=np.float32)
    p = bst.predict(X)
    order = np.argsort(-p)
    used_s, used_d = set(), set()
    nn = dict(nodes); ne = list(edges); nid = max(int(k) for k in nodes) + 1
    for i in order:
        if p[i] < th: break
        r = rows[i]
        if r['gap'] == 2 and not allow_gap2: continue
        s, d = r['s'], r['d']
        if s in used_s or d in used_d: continue
        used_s.add(s); used_d.add(d)
        if r['gap'] == 1:
            ne.append({'source_id': s, 'target_id': d, 'edge_link': round(float(p[i]), 4)}); stats['el_gap1'] += 1
        else:
            if r.get('drop_mid', 99.) <= snap_um and 'drop_xyz' in r:
                z, y, x = r['drop_xyz']
            else:
                a, b = nodes[s], nodes[d]
                z, y, x = [(float(a[k]) + float(b[k])) / 2 for k in 'zyx']
            nn[nid] = {'node_id': nid, 't': int(nodes[s]['t']) + 1, 'z': float(z), 'y': float(y), 'x': float(x), 'edge_link_node': 1}
            ne.append({'source_id': s, 'target_id': nid, 'edge_link': round(float(p[i]), 4)})
            ne.append({'source_id': nid, 'target_id': d, 'edge_link': round(float(p[i]), 4)})
            nid += 1; stats['el_gap2'] += 1
    return nn, ne, stats
