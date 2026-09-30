"""Independent re-implementation of 'relinefit' (re-smooth B5's output-filter linefit on the final P14 track structure).

B5 (cloud_baseline.linefit_smooth_output_graph) sets pos = (1-0.8)*orig + 0.8*linefit(orig over the +-2 fork-free chain) on the
REFERENCE-graph structure, before B5's lineage stages and every P-stage re-wire. For a node whose +-2 neighbourhood in the final
graph differs from the reference one:  new = cur + w * (f_final - f_ref),  f = (1-0.8)*orig + 0.8*linefit(orig over hood).
w = 1 - centroid_blend = 0.5 a priori (B5's centroid CNN stage already undid about half of any smoothing bias).

orig (the exact linefit input) reconstruction:
  exact=True : every reference node whose id is a fullgraph detection of the same frame takes the raw detection coordinate
               (no distance cap; the linefit itself can move a node > 3 um). Synthetic reference nodes (B5 gap inserts, not in the
               fullgraph) are solved exactly from the linefit equations ref_u = (1-W) o_u + W * sum_j a_uj o_j (sparse solve).
  exact=False: the r3_pl_relinefit / p15_post rule (raw only when |raw - ref| <= 3 um, reference coords otherwise).
Nodes that exist only in P14 (added after the output filter) enter the final fit at their current position and are not moved.
cen_aware: w = 1 for nodes whose current coords equal the reference coords (the centroid stage did not move them).
rand: control with the same shift lengths in a random direction. B5's 2 um collision rejection is re-applied.
Coordinates only: no node, edge or fork changes. GT-free."""
import sys, json, zlib
import builtins
from collections import defaultdict
import numpy as np
sys.path.insert(0, '/workspace/p56stage')
WANTS_META = True
REF = {'hold36': '/workspace/runs/b5f_hold36/working/reference_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/reference_graphs',
       'audit32': '/workspace/sync3/runs/b5f_audit32/working/reference_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/reference_graphs',
       't127b': '/workspace/sync3/runs/b5f_t127b/working/reference_graphs'}
S = np.array([1.625, .40625, .40625]); SHAPE = np.array([64, 256, 256]); W = 0.8; WIN = 2


def _struct(edges, t_of):
    pred = defaultdict(list); succ = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id'])
        if a in t_of and b in t_of and t_of[b] == t_of[a] + 1:
            succ[a].append(b); pred[b].append(a)
    return pred, succ


def _hood(n, pred, succ, have):
    h = [(0, n)]; c = n
    for k in range(1, WIN + 1):
        p = pred.get(c, [])
        if len(p) != 1: break
        c = p[0]
        if c not in have: break
        h.append((-k, c))
    c = n
    for k in range(1, WIN + 1):
        s = succ.get(c, [])
        if len(s) != 1: break
        c = s[0]
        if c not in have: break
        h.append((k, c))
    return tuple(sorted(h))


def _wts(h):
    """OLS line through (dt, x) evaluated at dt=0, as linear weights on the hood members."""
    d = np.array([dd for dd, _ in h], float); md = d.mean(); var = ((d - md) ** 2).sum()
    return 1.0 / len(d) + (0.0 - md) * (d - md) / var


def _f(k, h, orig):
    """B5 linefit output for node k with hood h and input positions orig."""
    if len(h) < 3: return orig[k]
    a = _wts(h); X = np.stack([orig[m] for _, m in h])
    return (1 - W) * orig[k] + W * (a[:, None] * X).sum(0)


def reconstruct_orig(rn, R_edges, raw, exact=True):
    t_ref = {k: int(v['t']) for k, v in rn.items()}
    pr, sr = _struct(R_edges, t_ref)
    refp = {k: np.array([v[c] for c in 'zyx'], float) for k, v in rn.items()}
    orig = {}; unk = []
    for k, v in rn.items():
        f = raw.get(k)
        if f is not None and f[0] == t_ref[k] and (exact or np.linalg.norm((f[1] - refp[k]) * S) <= 3.0): orig[k] = np.asarray(f[1], float)
        else:
            orig[k] = refp[k]
            if exact: unk.append(k)
    n_solved = 0
    if exact and unk:
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import spsolve
        idx = {k: i for i, k in enumerate(unk)}
        M = lil_matrix((len(unk), len(unk))); rhs = np.zeros((len(unk), 3))
        for k in unk:
            i = idx[k]; h = _hood(k, pr, sr, rn)
            if len(h) < 3: M[i, i] = 1.0; rhs[i] = refp[k]; continue
            a = _wts(h); rhs[i] = refp[k]; M[i, i] += (1 - W)
            for (d, m), am in zip(h, a):
                if m in idx: M[i, idx[m]] += W * am
                else: rhs[i] -= W * am * orig[m]
        M = M.tocsc()
        try:
            sol = np.column_stack([spsolve(M, rhs[:, c]) for c in range(3)])
            if np.all(np.isfinite(sol)) and np.abs(sol - np.stack([refp[k] for k in unk])).max() * 1.625 < 20.0:
                for k, i in idx.items(): orig[k] = sol[i]
                n_solved = len(unk)
        except Exception:
            pass
    # sanity over all reference nodes: does linefit(orig) on the reference structure reproduce the reference coords?
    ok = bad = 0
    for k in rn:
        h = _hood(k, pr, sr, rn)
        if np.linalg.norm((_f(k, h, orig) - refp[k]) * S) < 1e-3: ok += 1
        else: bad += 1
    return orig, pr, sr, {'rlf_sanity_ok': ok, 'rlf_sanity_bad': bad, 'rlf_unk': len(unk), 'rlf_solved': n_solved}


def apply(nodes, edges, name=None, set=None, fullgeff=None, zarr=None, w=0.5, exact=True, cen_aware=False, rand=False, minsep=2.0):
    from scipy.spatial import cKDTree
    from edge_link import load_full
    R = json.load(open(REF[set] + '/' + name + '.json'))
    rn = {int(k): v for k, v in R['nodes'].items()}
    fids, fT, fV, fE, fprob = load_full(fullgeff)
    raw = {int(i): (int(t), v) for i, t, v in zip(fids.tolist(), fT.tolist(), fV)}
    orig, pr, sr, st = reconstruct_orig(rn, R['edges'], raw, exact=exact)
    # nodes shared with the reference graph (same id, same frame); everything else enters the final fit at its current position
    shared = builtins.set(k for k, v in nodes.items() if k in rn and int(rn[k]['t']) == int(v['t']))
    orig_c = {k: (orig[k] if k in shared else np.array([v[c] for c in 'zyx'], float)) for k, v in nodes.items()}
    t_cur = {k: int(v['t']) for k, v in nodes.items()}
    pc, sc = _struct(edges, t_cur)
    new = {k: dict(v) for k, v in nodes.items()}; changed = builtins.set(); shifts = []; ncen = 0; ndiff = 0
    for k in shared:
        h_ref = _hood(k, pr, sr, rn); h_cur = _hood(k, pc, sc, nodes)
        if h_ref == h_cur: continue
        ndiff += 1
        cur = np.array([nodes[k][c] for c in 'zyx'], float)
        dlt = _f(k, h_cur, orig_c) - _f(k, h_ref, orig)
        ww = w
        if cen_aware and np.array_equal(cur, np.array([rn[k][c] for c in 'zyx'], float)): ww = 1.0; ncen += 1
        dlt = ww * dlt
        if not np.any(dlt): continue
        if rand:
            rg = np.random.default_rng(zlib.crc32(('%s_%d_%s' % (name, int(k), rand)).encode()))
            u = rg.normal(size=3); u /= np.linalg.norm(u)
            dlt = u * np.linalg.norm(dlt * S) / S
        q = cur + dlt
        if np.any(q < 0) or np.any(np.rint(q) >= SHAPE): continue
        new[k].update(dict(zip('zyx', map(float, q)))); changed.add(k); shifts.append(float(np.linalg.norm(dlt * S)))
    frames = defaultdict(list)
    for k, v in nodes.items(): frames[int(v['t'])].append(k)
    rej_total = 0
    for ns in frames.values():
        if len(ns) < 2 or not any(n in changed for n in ns): continue
        old = np.array([[nodes[n][c] for c in 'zyx'] for n in ns]) * S
        for _ in range(3):
            pp = np.array([[new[n][c] for c in 'zyx'] for n in ns]) * S
            rej = builtins.set()
            for i, j in cKDTree(pp).query_pairs(minsep):
                if np.linalg.norm(pp[i] - pp[j]) + 1e-5 < min(minsep, np.linalg.norm(old[i] - old[j])):
                    rej.update(n for n in (ns[i], ns[j]) if n in changed)
            if not rej: break
            for n in rej: new[n] = dict(nodes[n]); changed.discard(n)
            rej_total += len(rej)
    sh = np.array(shifts) if shifts else np.zeros(1)
    st.update({'rlf_diff': ndiff, 'rlf_moved': len(changed), 'rlf_rej': rej_total, 'rlf_cen1': ncen,
               'rlf_sh_gt05': int((sh > 0.5).sum()), 'rlf_sh_gt1': int((sh > 1.0).sum()), 'rlf_sh_gt2': int((sh > 2.0).sum())})
    return new, edges, st
