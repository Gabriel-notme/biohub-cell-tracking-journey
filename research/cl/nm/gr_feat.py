"""Edge features for every candidate of the union graph (gr_common.build_union). Used for training (gr_dump) and inference (gr_learned)."""
import gr_common as C
from collections import defaultdict
import numpy as np

PROV = ['motion_relinked', 'gap_closed', 'gap2_recovered', 'edge_link', 'relink', 'joint_event', 'safe_division', 'div_complete',
        'long_link', 'tbext', 'term_trim']
FEATS = ['p15', 'prov', 'ep15', 'fe', 'ilp_e', 'a_in', 'b_in', 'a_ilp', 'b_ilp', 'a_syn', 'b_syn', 'dist', 'dz', 'dxy',
         'a_out', 'a_indeg', 'b_indeg', 'b_out', 'a_hist', 'b_fut', 'sp_a', 'dpred_a', 'cos_a', 'sp_b', 'dpred_b', 'cos_b',
         'rank_a', 'n_a', 'gap_a', 'rank_b', 'n_b', 'gap_b', 'fe_oth_a', 'fe_oth_b', 'cur_a_d', 'cur_a_fe', 'cur_b_d', 'cur_b_fe',
         'nn_a', 'nn_b', 'nnp_a', 'nnp_b', 'dens_a', 'dens_b', 'z', 'tt', 'a_fin', 'b_fout', 'a_forkchild', 'b_fork', 'src_rad']


def ilp_sets(set_, name):
    import os
    p = '/workspace/cl/ideas/r4_ilp/%s/%s.npz' % (set_, name)
    if not os.path.exists(p): return set(), set()
    z = np.load(p)
    return set(z['nodes'].tolist()), set(zip(z['src'].tolist(), z['dst'].tolist()))


def features(U, nodes, edges, full, ilp=(set(), set())):
    from scipy.spatial import cKDTree
    ids, t, pos, inp = U['ids'], U['t'], U['pos'], U['inp']
    ix = U['ix']; cand = U['cand']; n = len(ids)
    fids, fT, fV, fE, fprob = full
    infull = np.zeros(n, bool)
    for fi in fids.tolist():
        j = ix.get(int(fi))
        if j is not None: infull[j] = True
    ilp_nodes, ilp_edges = ilp
    ilpn = np.array([int(x) in ilp_nodes for x in ids.tolist()])
    par = {}; succ = defaultdict(list); eprov = {}; eprob = {}
    for e in edges:
        a, b = ix[int(e['source_id'])], ix[int(e['target_id'])]
        if t[b] != t[a] + 1: continue
        par[b] = a; succ[a].append(b)
        pv = 0
        for k, nm in enumerate(PROV, 1):
            if nm in e: pv = k; break
        eprov[(a, b)] = pv
        try: eprob[(a, b)] = float(e.get('edge_prob', -1.0))
        except (TypeError, ValueError): eprob[(a, b)] = -1.0
    fin = np.full(n, -1.0); fout = np.full(n, -1.0)
    for (a0, b0), p in zip(fE.tolist(), fprob.tolist()):
        a, b = ix.get(int(a0)), ix.get(int(b0))
        if a is None or b is None: continue
        fin[b] = max(fin[b], p); fout[a] = max(fout[a], p)
    forkp = {a for a, bs in succ.items() if len(bs) >= 2}
    forkc = {b for a in forkp for b in succ[a]}

    def back(x, lim=30):
        k = 0
        while x in par and k < lim: x = par[x]; k += 1
        return k

    def fwd(x, lim=30):
        k = 0
        while len(succ.get(x, [])) == 1 and k < lim: x = succ[x][0]; k += 1
        return k
    hist = {}; fut = {}
    frames = defaultdict(list); pframes = defaultdict(list)
    for i in range(n):
        frames[int(t[i])].append(i)
        if inp[i]: pframes[int(t[i])].append(i)
    trees = {tt: (np.array(L), cKDTree(pos[L])) for tt, L in frames.items()}
    ptrees = {tt: (np.array(L), cKDTree(pos[L])) for tt, L in pframes.items() if L}

    def nn_other(i, tr):
        if int(t[i]) not in tr: return 99.0
        L, T_ = tr[int(t[i])]
        k = min(2, len(L)); d, j = T_.query(pos[i], k=k)
        d = np.atleast_1d(d); j = np.atleast_1d(j)
        for dd, jj in zip(d, j):
            if L[jj] != i: return float(dd)
        return 99.0
    nnU = {}; nnP = {}; dens = {}
    ca = defaultdict(list); cb = defaultdict(list)
    keys = list(cand)
    D = np.array([np.linalg.norm(pos[b] - pos[a]) for a, b in keys]) if keys else np.zeros(0)
    for k, (a, b) in enumerate(keys): ca[a].append((D[k], cand[(a, b)]['fe'], b)); cb[b].append((D[k], cand[(a, b)]['fe'], a))
    for d_ in (ca, cb):
        for kk in d_: d_[kk].sort()
    Tm = max(1, int(t.max()))
    X = np.zeros((len(keys), len(FEATS)), np.float32)
    for k, (a, b) in enumerate(keys):
        r = cand[(a, b)]
        for x in (a, b):
            if x not in nnU:
                nnU[x] = nn_other(x, trees); nnP[x] = nn_other(x, ptrees)
                L, T_ = trees[int(t[x])]; dens[x] = len(T_.query_ball_point(pos[x], 8.0)) - 1
        if a not in hist: hist[a] = back(a)
        if b not in fut: fut[b] = fwd(b)
        disp = pos[b] - pos[a]; dist = float(D[k])
        va = pos[a] - pos[par[a]] if a in par else None
        vb = pos[succ[b][0]] - pos[b] if len(succ.get(b, [])) == 1 else None
        la = ca[a]; lb = cb[b]
        ra = [x[2] for x in la].index(b); rb = [x[2] for x in lb].index(a)
        ga = (la[1][0] - dist) if (ra == 0 and len(la) > 1) else (dist - la[0][0])
        gb = (lb[1][0] - dist) if (rb == 0 and len(lb) > 1) else (dist - lb[0][0])
        foa = max([x[1] for x in la if x[2] != b], default=-1.0); fob = max([x[1] for x in lb if x[2] != a], default=-1.0)
        cur_a = [c for c in succ.get(a, []) if c != b]; cur_b = par.get(b)
        cur_b = cur_b if (cur_b is not None and cur_b != a) else None
        X[k] = [r['p15'], eprov.get((a, b), -1), eprob.get((a, b), -1.0), r['fe'], int((int(ids[a]), int(ids[b])) in ilp_edges),
                int(inp[a]), int(inp[b]), int(ilpn[a]), int(ilpn[b]), int(not infull[a]), int(not infull[b]), dist, abs(disp[0]),
                float(np.linalg.norm(disp[1:])), len(succ.get(a, [])), int(a in par), int(b in par), len(succ.get(b, [])), hist[a], fut[b],
                float(np.linalg.norm(va)) if va is not None else -1.0,
                float(np.linalg.norm(pos[b] - pos[a] - va)) if va is not None else -1.0,
                float(va @ disp / (np.linalg.norm(va) * dist + 1e-6)) if va is not None else 0.0,
                float(np.linalg.norm(vb)) if vb is not None else -1.0,
                float(np.linalg.norm(pos[a] - (pos[b] - vb))) if vb is not None else -1.0,
                float(vb @ disp / (np.linalg.norm(vb) * dist + 1e-6)) if vb is not None else 0.0,
                ra, len(la), ga, rb, len(lb), gb, foa, fob,
                float(np.linalg.norm(pos[cur_a[0]] - pos[a])) if cur_a else -1.0,
                (cand.get((a, cur_a[0]), {}).get('fe', -1.0)) if cur_a else -2.0,
                float(np.linalg.norm(pos[b] - pos[cur_b])) if cur_b is not None else -1.0,
                (cand.get((cur_b, b), {}).get('fe', -1.0)) if cur_b is not None else -2.0,
                nnU[a], nnU[b], nnP[a], nnP[b], dens[a], dens[b], float(pos[a][0]), t[a] / Tm, fin[a], fout[b],
                int(a in forkc), int(b in forkc), int(r['src'] == 'rad')]
    return keys, X
