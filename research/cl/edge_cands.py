"""Free-end linking candidates on P3 graphs: (s end at t) -> (d start at t+gap), gap in {1,2}, with features + GT labels."""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
import zarr
S = np.array([1.625, .40625, .40625])
SETS = {'hold36': ('/workspace/hold36.txt', '/workspace/runs/p3_hold36/graphs', '/workspace/runs/fullgraph_hold36'),
        'prev4': ('/workspace/preview4.txt', '/workspace/runs/p3_prev4/graphs', '/workspace/runs/fullgraph_prev4'),
        'audit32': ('/workspace/audit32.txt', '/workspace/sync3/runs/p3_audit32/graphs', '/workspace/sync3/runs/fullgraph_audit32'),
        't127a': ('/workspace/t127a.txt', '/workspace/sync4/runs/fin_p3_t127a/graphs', '/workspace/sync4/runs/fullgraph_t127a'),
        't127b': ('/workspace/t127b.txt', '/workspace/sync3/runs/fin_p3_t127b/graphs', '/workspace/sync3/runs/fullgraph_t127b')}
OUT = Path('/workspace/cl/ecands')
RMAX = {1: 14.0, 2: 18.0}


def features(nodes, edges, fullgeff):
    succ = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    trees = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in frames.items()}
    fedge = {}; dtrees = {}
    if fullgeff is not None and Path(fullgeff).exists():
        fg = zarr.open_group(str(fullgeff), mode='r')
        fids = np.asarray(fg['nodes/ids'][:]).astype(np.int64); fT = np.asarray(fg['nodes/props/t/values'][:]).astype(int)
        fP = np.stack([np.asarray(fg['nodes/props/%s/values' % k][:]) for k in 'zyx'], 1) * S
        fE = np.asarray(fg['edges/ids'][:]).astype(np.int64); fprob = np.asarray(fg['edges/props/edge_prob/values'][:])
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


def label(name, nodes, edges, rows):
    import evalx
    K = evalx.K
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    inv = {v: k for k, v in mapping.items()}
    from tracking_cellmot.metrics import evaluate
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ea = gt.edge_attrs(); gsucc = defaultdict(list); gpar = {}
    for a, b in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gsucc[int(a)].append(int(b)); gpar[int(b)] = int(a)
    for r in rows:
        gs, gd = p2g.get(r['s']), p2g.get(r['d'])
        valid = (gs is not None and len(gsucc.get(gs, [])) > 0) or (gd is not None and gd in gpar)
        if r['gap'] == 1: ok = gs is not None and gd is not None and gd in gsucc.get(gs, [])
        else: ok = gs is not None and gd is not None and any(gd in gsucc.get(x, []) for x in gsucc.get(gs, []))
        r['lab'] = 'P' if ok else ('N' if valid else 'U')


def job(args):
    s, name, gdir, fdir = args
    d = json.load(open(Path(gdir) / (name + '.json')))
    nodes = {int(k): v for k, v in d['nodes'].items()}; edges = d['edges']
    rows = features(nodes, edges, Path(fdir) / (name + '.geff'))
    label(name, nodes, edges, rows)
    for r in rows: r['set'] = s; r['movie'] = name
    (OUT / ('%s__%s.json' % (s, name))).write_text(json.dumps(rows))
    return s, name, len(rows), sum(r['lab'] == 'P' for r in rows), sum(r['lab'] == 'N' for r in rows)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    jobs = []
    for s, (lst, g, f) in SETS.items():
        for n in [l.strip() for l in open(lst) if l.strip()]: jobs.append((s, n, g, f))
    from collections import Counter
    tot = Counter()
    with Pool(24) as pool:
        for s, n, a, p, ng in pool.imap_unordered(job, jobs):
            tot[(s, 'rows')] += a; tot[(s, 'P')] += p; tot[(s, 'N')] += ng
    for k in sorted(tot): print(k, tot[k])
    print('EDGE_CANDS_DONE')
