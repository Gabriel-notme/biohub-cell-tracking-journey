"""P15 post-steps (after the P14 steps): 'tbext' then 'relinefit'. No learned parameters, no model.

tbext: B5 keeps the ILP solution in the pre-ILP candidate graph (fullgraph), then relinks by per-frame motion Hungarian and drops
tracks shorter than 6 nodes, so real cells lose ILP-supported heads/tails whose detections exist. For every track START (END)
with >= minlen nodes, follow the fullgraph parent (child) edge with prob >= pmin into a DROPPED detection (fullgraph id not in the
graph) of the adjacent frame, provided no kept node lies within dup um there; add node + dt=1 edge and repeat. join: if the added
node's own fullgraph neighbour on the far side is a kept free END (START), add that edge and stop (closes the gap).
Never creates forks or merges. Derived from ideas/r3_tbext.py (mode 'edge').

relinefit: coordinates only (no node/edge/fork changes).

B5 smooths node coordinates inside its output filter with a line fit (weight 0.8, +-2 frames along fork-free chains) on the
REFERENCE-graph structure, i.e. before B5's lineage stages and before every P-stage re-wire (div_complete, dfork, relink,
edge_link, term_trim join, long_link). A node whose final +-2 track neighbourhood differs from the one used then was smoothed
with neighbours that may now belong to another cell. Re-smooth it on the final structure:
    new = cur + w * (fit_final - fit_ref),   fit = (1 - 0.8) * orig + 0.8 * linefit(orig over the +-2 neighbourhood)
with orig = pre-smoothing coordinates (raw pre-ILP detection coordinates for id-aligned detections, reference coordinates
otherwise; nodes added after the output filter enter the fit at their current position). w = 1 - centroid_blend = 0.5 because
B5's centroid stage (blend 0.5) already corrected about half of the smoothing bias. B5's 2 um collision rule is re-applied.
Derived from ideas/r3_pl_relinefit.py (src='final', w=0.5, all categories)."""
from collections import defaultdict
import json
import numpy as np

S = np.array([1.625, .40625, .40625]); W = 0.8


def _struct(edges, t_of):
    pred = defaultdict(list); succ = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id'])
        if a in t_of and b in t_of and t_of[b] == t_of[a] + 1:
            succ[a].append(b); pred[b].append(a)
    return pred, succ


def _hood(n, pred, succ, have):
    h = [(0, n)]; c = n
    for k in range(1, 3):
        p = pred.get(c, [])
        if len(p) != 1: break
        c = p[0]
        if c not in have: break
        h.append((-k, c))
    c = n
    for k in range(1, 3):
        s = succ.get(c, [])
        if len(s) != 1: break
        c = s[0]
        if c not in have: break
        h.append((k, c))
    return tuple(sorted(h))


def _fit(h, orig):
    dts = np.array([d for d, _ in h], float); X = np.stack([orig[m] for _, m in h])
    md = dts.mean(); mx = X.mean(0); var = ((dts - md) ** 2).sum()
    slope = ((dts - md)[:, None] * (X - mx)).sum(0) / var
    return mx - slope * md


def _rpos(v):
    return np.array([max(0, int(round(float(v[k])))) for k in 'zyx'], float) * S


def tbext(nodes, edges, fullgeff, K=100, minlen=5, pmin=0.8, dup=3.5, join=True):
    from scipy.spatial import cKDTree
    from edge_link import load_full
    fids, fT, fV, fE, fprob = load_full(fullgeff)
    fidx = {int(i): j for j, i in enumerate(fids.tolist())}
    fpar = defaultdict(list); fch = defaultdict(list)
    for (a, b), p in zip(fE.tolist(), fprob.tolist()):
        fpar[int(b)].append((float(p), int(a))); fch[int(a)].append((float(p), int(b)))
    par = {}; succ = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); par[b] = a; succ[a].append(b)
    ts = [int(v['t']) for v in nodes.values()]; T0, T1 = min(ts), max(ts)
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    trees = {t: cKDTree(np.stack([_rpos(nodes[n]) for n in ns])) for t, ns in byt.items()}

    def chain_len(n, down):
        L = 1; c = n
        while L < minlen:
            nx = succ.get(c, []) if down else ([par[c]] if c in par else [])
            if len(nx) != 1: break
            c = nx[0]; L += 1
        return L
    new_nodes = dict(nodes); new_edges = list(edges); used = set(); added = defaultdict(int)

    def free(j, t):
        if j in used or int(fids[j]) in new_nodes: return False
        if dup is not None and t in trees:
            if trees[t].query_ball_point(_rpos({'z': fV[j][0], 'y': fV[j][1], 'x': fV[j][2]}), dup): return False
        return True

    def step(cur, t_next, backward):
        cands = fpar.get(cur, []) if backward else fch.get(cur, [])
        cands = sorted([(p, x) for p, x in cands if p >= pmin and x in fidx and int(fT[fidx[x]]) == t_next], reverse=True)
        for p, x in cands:
            if free(fidx[x], t_next): return fidx[x]
        return None
    seeds = [(n, True) for n in nodes if n not in par and T0 < int(nodes[n]['t']) <= T0 + K and chain_len(n, True) >= minlen]
    seeds += [(n, False) for n in nodes if not succ.get(n) and T1 - K <= int(nodes[n]['t']) < T1 and chain_len(n, False) >= minlen]
    got_child = set(); got_par = set()
    for n, backward in seeds:
        if backward and n in got_par: continue
        if (not backward) and n in got_child: continue
        cur = n
        while True:
            t = int(new_nodes[cur]['t']); tn = t - 1 if backward else t + 1
            if tn < T0 or tn > T1: break
            j = step(cur, tn, backward)
            if j is None: break
            used.add(j); nid = int(fids[j])
            new_nodes[nid] = {'node_id': nid, 't': tn, 'z': float(fV[j][0]), 'y': float(fV[j][1]), 'x': float(fV[j][2]), 'tb_ext': 1}
            new_edges.append({'source_id': nid, 'target_id': cur, 'tb_ext': 1} if backward else {'source_id': cur, 'target_id': nid, 'tb_ext': 1})
            if backward: got_par.add(cur)
            else: got_child.add(cur)
            added['start' if backward else 'end'] += 1
            cur = nid
            if join:
                far = fpar.get(cur, []) if backward else fch.get(cur, [])
                tf = tn - 1 if backward else tn + 1
                hit = None
                for p, x in sorted(far, reverse=True):
                    if p < pmin or x not in nodes or int(nodes[x]['t']) != tf: continue
                    if backward and not succ.get(x) and x not in got_child: hit = x; break
                    if (not backward) and x not in par and x not in got_par: hit = x; break
                if hit is not None:
                    if backward: new_edges.append({'source_id': hit, 'target_id': cur, 'tb_join': 1}); got_child.add(hit)
                    else: new_edges.append({'source_id': cur, 'target_id': hit, 'tb_join': 1}); got_par.add(hit)
                    added['join'] += 1
                    break
    return new_nodes, new_edges, {'tb_add_start': added['start'], 'tb_add_end': added['end'], 'tb_join': added['join'], 'tb_seeds': len(seeds)}


def relinefit(nodes, edges, refgraph, fullgeff, shape=(64, 256, 256), w=0.5, minsep=2.0):
    from scipy.spatial import cKDTree
    from edge_link import load_full
    SHAPE = np.array(shape)
    R = json.loads(open(refgraph).read())
    rn = {int(k): v for k, v in R['nodes'].items()}
    fids, fT, fV, fE, fprob = load_full(fullgeff)
    raw = {int(i): (int(t), v) for i, t, v in zip(fids.tolist(), fT.tolist(), fV)}
    orig = {}
    for k, v in rn.items():
        f = raw.get(k)
        rp = np.array([v[c] for c in 'zyx'], float)
        if f is not None and f[0] == int(v['t']) and np.linalg.norm((f[1] - rp) * S) <= 3.0: orig[k] = np.asarray(f[1], float)
        else: orig[k] = rp
    t_ref = {k: int(v['t']) for k, v in rn.items()}
    pr, sr = _struct(R['edges'], t_ref)
    ok = bad = 0  # sanity: linefit on the reference structure should reproduce the reference coordinates
    for k in list(rn)[::50]:
        if k not in raw: continue
        h = _hood(k, pr, sr, orig)
        rp = np.array([rn[k][c] for c in 'zyx'], float)
        pos = orig[k] if len(h) < 3 else (1 - W) * orig[k] + W * _fit(h, orig)
        if np.linalg.norm((pos - rp) * S) < 1e-3: ok += 1
        else: bad += 1
    t_cur = {k: int(v['t']) for k, v in nodes.items()}
    pc, sc = _struct(edges, t_cur)
    orig_c = dict(orig)
    for k, v in nodes.items():
        if k not in orig_c: orig_c[k] = np.array([v[c] for c in 'zyx'], float)
    have_c = set(k for k in nodes if k in orig_c)
    new = {k: dict(v) for k, v in nodes.items()}; changed = set(); shifts = []
    for k in nodes:
        if k not in orig: continue
        h_ref = _hood(k, pr, sr, orig); h_cur = _hood(k, pc, sc, have_c)
        if h_ref == h_cur: continue
        f_ref = orig[k] if len(h_ref) < 3 else (1 - W) * orig[k] + W * _fit(h_ref, orig)
        f_cur = orig[k] if len(h_cur) < 3 else (1 - W) * orig[k] + W * _fit(h_cur, orig_c)
        dlt = w * (f_cur - f_ref)
        if not np.any(dlt): continue
        q = np.array([nodes[k][c] for c in 'zyx'], float) + dlt
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
            rej = set()
            for i, j in cKDTree(pp).query_pairs(minsep):
                if np.linalg.norm(pp[i] - pp[j]) + 1e-5 < min(minsep, np.linalg.norm(old[i] - old[j])):
                    rej.update(n for n in (ns[i], ns[j]) if n in changed)
            if not rej: break
            for n in rej: new[n] = dict(nodes[n]); changed.discard(n)
            rej_total += len(rej)
    sh = np.array(shifts) if shifts else np.zeros(1)
    return new, edges, {'rlf_moved': len(changed), 'rlf_rej': rej_total, 'rlf_sanity_ok': ok, 'rlf_sanity_bad': bad, 'rlf_sh_gt1': int((sh > 1.0).sum())}
