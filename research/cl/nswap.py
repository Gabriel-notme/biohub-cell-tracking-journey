"""Node substitution: replace a track node n by a dropped pre-ILP detection c at the same frame (the ILP kept the wrong one of two
nearby detections). Candidate generation + features (graph-only; pre-ILP candidate graph gives edge probabilities).
features(nodes, edges, full) -> list of (n, c_index, feature_vector)."""
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
S = np.array([1.625, 0.40625, 0.40625])
FEATS = ['d_nc', 'fe_pn', 'fe_nq', 'fe_pc', 'fe_cq', 'has_p', 'has_q', 'd_pn', 'd_nq', 'd_pc', 'd_cq', 'mid_n', 'mid_c', 'dz_nc', 'dyx_nc',
         'c_deg_in', 'c_deg_out', 'c_maxin', 'c_maxout', 'n_raw_shift', 'near_other', 'crowd_n', 'crowd_c', 'track_len', 'z_n', 'z_c',
         'fe_pn_rank', 'fe_pc_rank', 'c_frag_len', 'n_best_in', 'n_best_out']


def load_full(fullgeff):
    import zarr
    fg = zarr.open_group(str(fullgeff), mode='r')
    fids = np.asarray(fg['nodes/ids'][:]).astype(np.int64); fT = np.asarray(fg['nodes/props/t/values'][:]).astype(int)
    fV = np.stack([np.asarray(fg['nodes/props/%s/values' % k][:]) for k in 'zyx'], 1).astype(float)
    fE = np.asarray(fg['edges/ids'][:]).astype(np.int64); fprob = np.asarray(fg['edges/props/edge_prob/values'][:]).astype(float)
    return fids, fT, fV, fE, fprob


def candidates(nodes, edges, full, R=12.0, clear=4.0):
    fids, fT, fV, fE, fprob = full
    fP = fV * S
    fid2i = {int(f): i for i, f in enumerate(fids.tolist())}
    fin = defaultdict(list); fout = defaultdict(list)
    fe = {}
    for (a, b), p in zip(fE.tolist(), fprob.tolist()):
        fe[(a, b)] = p; fout[a].append((b, p)); fin[b].append((a, p))
    ch = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); ch[a].append(b); par[b] = a
    pos = {n: np.array([v['z'], v['y'], v['x']], float) * S for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    ktree = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in frames.items()}
    drop = defaultdict(list)
    for i, (f, t) in enumerate(zip(fids.tolist(), fT.tolist())):
        if f not in nodes: drop[t].append(i)
    dtree = {t: (ix, cKDTree(fP[ix])) for t, ix in drop.items() if ix}
    # dropped fragment chain length through c (following best dropped-only fullgraph edges)
    dropped = set(fids[i] for ix in drop.values() for i in ix)

    def frag_len(f):
        k = 1; x = f
        for _ in range(10):
            nx = [b for b, p in fout.get(x, []) if b in dropped and p > 0.5]
            if not nx: break
            x = nx[0]; k += 1
        x = f
        for _ in range(10):
            pv = [a for a, p in fin.get(x, []) if a in dropped and p > 0.5]
            if not pv: break
            x = pv[0]; k += 1
        return k

    def tlen(n):
        k = 1; x = n
        while x in par and k < 50: x = par[x]; k += 1
        x = n
        while len(ch.get(x, [])) == 1 and k < 100: x = ch[x][0]; k += 1
        return k
    rows = []
    for t, (ix, tr) in dtree.items():
        if t not in ktree: continue
        ns, ntr = ktree[t]
        for n in ns:
            if len(ch.get(n, [])) > 1: continue  # never touch forks
            p = par.get(n); q = ch[n][0] if len(ch.get(n, [])) == 1 else None
            if p is not None and len(ch.get(p, [])) > 1: continue  # never touch fork children
            if p is None and q is None: continue
            for j in tr.query_ball_point(pos[n], R):
                ci = ix[j]; c = int(fids[ci]); pc = fP[ci]
                # c must be clear of every other kept node (else it is a duplicate of another cell)
                near = ntr.query_ball_point(pc, clear)
                if any(ns[k] != n for k in near): continue
                d_nc = float(np.linalg.norm(pos[n] - pc))
                f = {}
                f['d_nc'] = d_nc
                f['fe_pn'] = fe.get((p, n), 0.) if p is not None else -1
                f['fe_nq'] = fe.get((n, q), 0.) if q is not None else -1
                f['fe_pc'] = fe.get((p, c), 0.) if p is not None else -1
                f['fe_cq'] = fe.get((c, q), 0.) if q is not None else -1
                f['has_p'] = int(p is not None); f['has_q'] = int(q is not None)
                f['d_pn'] = float(np.linalg.norm(pos[p] - pos[n])) if p is not None else -1
                f['d_nq'] = float(np.linalg.norm(pos[q] - pos[n])) if q is not None else -1
                f['d_pc'] = float(np.linalg.norm(pos[p] - pc)) if p is not None else -1
                f['d_cq'] = float(np.linalg.norm(pos[q] - pc)) if q is not None else -1
                if p is not None and q is not None:
                    m = (pos[p] + pos[q]) / 2; f['mid_n'] = float(np.linalg.norm(pos[n] - m)); f['mid_c'] = float(np.linalg.norm(pc - m))
                else:
                    f['mid_n'] = f['mid_c'] = -1
                f['dz_nc'] = float(abs(pos[n][0] - pc[0])); f['dyx_nc'] = float(np.linalg.norm(pos[n][1:] - pc[1:]))
                f['c_deg_in'] = len(fin.get(c, [])); f['c_deg_out'] = len(fout.get(c, []))
                f['c_maxin'] = max([pp for a, pp in fin.get(c, [])], default=0.); f['c_maxout'] = max([pp for b, pp in fout.get(c, [])], default=0.)
                f['n_raw_shift'] = float(np.linalg.norm(fP[fid2i[n]] - pos[n])) if n in fid2i else -1
                f['near_other'] = float(sorted(ntr.query(pc, k=min(2, len(ns)))[0])[-1]) if len(ns) > 1 else 99.
                f['crowd_n'] = len(ntr.query_ball_point(pos[n], 10.)); f['crowd_c'] = len(tr.query_ball_point(pc, 10.))
                f['track_len'] = tlen(n); f['z_n'] = float(pos[n][0]); f['z_c'] = float(pc[0])
                ins = sorted([pp for a, pp in fin.get(n, [])], reverse=True)
                f['fe_pn_rank'] = float(ins.index(f['fe_pn'])) if f['fe_pn'] in ins else -1
                insc = sorted([pp for a, pp in fin.get(c, [])], reverse=True)
                f['fe_pc_rank'] = float(insc.index(f['fe_pc'])) if f['fe_pc'] in insc else -1
                f['c_frag_len'] = frag_len(c)
                f['n_best_in'] = ins[0] if ins else 0.; outs = [pp for b, pp in fout.get(n, [])]; f['n_best_out'] = max(outs, default=0.)
                rows.append((n, ci, [f[k] for k in FEATS]))
    return rows


def apply(nodes, edges, full, model_path, th=0.5):
    """Greedy substitution by predicted probability; each n and c used once. Node ids kept (only positions change)."""
    import lgb_np
    rows = candidates(nodes, edges, full)
    if not rows: return nodes, edges, {'ns_candidates': 0, 'ns_applied': 0}
    X = np.array([r[2] for r in rows], float)
    pr = lgb_np.load(model_path).predict(X)
    fids, fT, fV, fE, fprob = full
    order = np.argsort(-pr); used_n = set(); used_c = set(); nodes = dict(nodes); k = 0
    for i in order:
        if pr[i] < th: break
        n, ci, _ = rows[i]
        if n in used_n or ci in used_c: continue
        v = dict(nodes[n]); v['z'], v['y'], v['x'] = float(fV[ci, 0]), float(fV[ci, 1]), float(fV[ci, 2]); v['nswap'] = 1
        nodes[n] = v; used_n.add(n); used_c.add(ci); k += 1
    return nodes, edges, {'ns_candidates': len(rows), 'ns_applied': k}
