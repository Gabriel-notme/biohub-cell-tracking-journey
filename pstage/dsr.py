"""Dropped-sister recovery: give a single-child track a second daughter taken from
detector candidates that the base ILP dropped, when the frozen b1 fork head is confident."""
import numpy as np, zarr
from collections import defaultdict
from pathlib import Path

def load_full(path):
    g = zarr.open_group(str(path), mode='r')
    T = np.asarray(g['nodes/props/t/values'][:]).astype(int)
    V = np.stack([np.asarray(g['nodes/props/%s/values' % k][:]) for k in 'zyx'], 1).astype(np.float32)
    return T, V

def recover(er, name, nodes, edges, full_path, th=0.95, max_pb=13.0, max_ab=20.0, sep=4.0, fut_um=6.0, link_future=True):
    from scipy.spatial import cKDTree
    from cell_event import SCALE, chain, fork_geometry
    out = defaultdict(list); prev = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); out[s].append(d); prev[d] = s
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    trees = {t: cKDTree(np.array([pos[n] for n in ns])) for t, ns in frames.items() if ns}
    FT, FV = load_full(full_path); FP = FV * SCALE
    dropped = {}
    for t in np.unique(FT):
        idx = np.where(FT == t)[0]
        if int(t) in trees:
            d, _ = trees[int(t)].query(FP[idx]); idx = idx[d > sep]
        if len(idx): dropped[int(t)] = idx
    dtrees = {t: cKDTree(FP[ix]) for t, ix in dropped.items()}
    starts = defaultdict(list)
    for n, v in nodes.items():
        if n not in prev: starts[int(v['t'])].append(n)
    stree = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in starts.items() if ns}
    cands = []
    for t, ns in frames.items():
        if t + 1 not in dtrees: continue
        ix = dropped[t + 1]
        for p in ns:
            ch = out.get(p, [])
            if len(ch) != 1: continue
            a = ch[0]
            for j in dtrees[t + 1].query_ball_point(pos[p], max_pb):
                fi = int(ix[j])
                if np.linalg.norm(FP[fi] - pos[a]) > max_ab: continue
                cands.append((p, a, fi))
    stats = {'dsr_candidates': len(cands), 'dsr_added': 0, 'dsr_future_links': 0}
    if not cands: return nodes, edges, stats
    NEW0 = 10 ** 9; newid = {}
    for p, a, fi in cands:
        if fi not in newid: newid[fi] = NEW0 + len(newid)
    sub = {n: nodes[n] for c in cands for n in c[:2]}
    for fi, nid in newid.items():
        sub[nid] = {'t': int(FT[fi]), 'z': float(FV[fi][0]), 'y': float(FV[fi][1]), 'x': float(FV[fi][2])}
    ids, emb = er.embeddings(name, sub); lookup = {n: i for i, n in enumerate(ids)}
    fut = {}
    for fi in newid:
        t2 = int(FT[fi]) + 1; f = None
        if link_future and t2 in stree:
            ns2, tr2 = stree[t2]; d, j = tr2.query(FP[fi])
            if d <= fut_um: f = ns2[int(j)]
        fut[fi] = f
    tri = [(p, a, newid[fi]) for p, a, fi in cands]
    fg = []
    for p, a, fi in cands:
        b = np.vstack([FP[fi][None], chain(fut[fi], out, pos)]) if fut[fi] is not None else FP[fi][None]
        fg.append(fork_geometry(chain(p, prev, pos), chain(a, out, pos), b))
    prob = er.score('fork', tri, fg, emb, lookup)
    order = np.argsort(-prob)
    used_p, used_f, used_s = set(), set(), set()
    nn = dict(nodes); ne = list(edges); nid = max(int(k) for k in nodes) + 1
    for i in order:
        if prob[i] < th: break
        p, a, fi = cands[i]
        if p in used_p or fi in used_f: continue
        f = fut[fi]
        if f is not None and f in used_s: f = None
        used_p.add(p); used_f.add(fi)
        nn[nid] = {'node_id': nid, 't': int(FT[fi]), 'z': float(FV[fi][0]), 'y': float(FV[fi][1]), 'x': float(FV[fi][2])}
        ne.append({'source_id': p, 'target_id': nid, 'dsr': 1})
        if f is not None:
            ne.append({'source_id': nid, 'target_id': f, 'dsr': 1}); used_s.add(f); stats['dsr_future_links'] += 1
        nid += 1; stats['dsr_added'] += 1
    return nn, ne, stats
