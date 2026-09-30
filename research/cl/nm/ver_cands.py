"""ver_cands: enumerate candidate ADDITIONS at P15 free track termini (GT-labelled, READ-ONLY on P15 outputs).
Walk family: from every free END (t<T1) / free START (t>T0) follow the best pre-ILP dt=1 edge (prob >= PW) into a DROPPED
detection (fullgraph id not in the graph, > dup um from every kept/added node in its frame), repeat up to KMAX steps; at each
added node check a JOIN: its far-side pre-ILP neighbour (prob >= PW) is a free terminus of the opposite kind -> join edge, stop.
Near family: terminus without any pre-ILP step: nearest dropped detection within RN um of the (velocity-extrapolated) position.
Labels (per new edge, official validity rule): TP (edge maps to a GT edge whose GT endpoint is unmatched in P15),
FP (edge valid = source GT node has out-degree>0 or target GT node has in-degree>0, but not TP), NE (not evaluable).
Also a census of every P15 FN edge by type, and of evaluable termini whose GT continuation is unmatched (reachability).
Writes /workspace/cl/nm/ver_cands/<set>/<movie>.json and /workspace/cl/nm/ver_census.json"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p56stage')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
OUT = '/workspace/cl/nm/ver_cands'
PW, KMAX, DUP, RN = 0.05, 12, 3.5, 6.0


def rp(v):
    return np.array([max(0, int(round(float(x)))) for x in v], float) * S


def job(f):
    try:
        import numcodecs.blosc
        numcodecs.blosc.use_threads = False
    except Exception:
        pass
    import evalx
    from tracking_cellmot.metrics import evaluate
    from edge_link import load_full
    from scipy.spatial import cKDTree
    K = evalx.K
    name = Path(f).stem
    st = [s for s in SETS if '_%s/' % s in f][0]
    nodes, edges = evalx.load_graph_json(f)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: p for p, g in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {}; gt_t = {}
    for i, t, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['t'].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list()):
        gpos[int(i)] = np.array([z, y, x], float) * S; gt_t[int(i)] = int(t)
    ea = gt.edge_attrs()
    gsucc = defaultdict(list); gpar = defaultdict(list)
    for s_, d_ in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gsucc[int(s_)].append(int(d_)); gpar[int(d_)].append(int(s_))
    gunm = defaultdict(list)
    for g, t in gt_t.items():
        if g not in g2p: gunm[t].append(g)
    gut = {t: (v, cKDTree(np.stack([gpos[g] for g in v]))) for t, v in gunm.items()}

    succ = defaultdict(list); par = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b].append(a)
    ts = [int(v['t']) for v in nodes.values()]; T0, T1 = min(ts), max(ts)
    ppos = {n: rp([v['z'], v['y'], v['x']]) for n, v in nodes.items()}
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    ptree = {t: cKDTree(np.stack([ppos[n] for n in ns])) for t, ns in byt.items()}
    fids, fT, fV, fE, fprob = load_full(FULL[st] + '/' + name + '.geff')
    fidx = {int(i): j for j, i in enumerate(fids.tolist())}
    fpos = [rp(v) for v in fV]
    fch = defaultdict(list); fpa = defaultdict(list)
    for (a, b), p in zip(fE.tolist(), fprob.tolist()):
        a, b = int(a), int(b)
        if a in fidx and b in fidx and fT[fidx[b]] == fT[fidx[a]] + 1: fch[a].append((float(p), b)); fpa[b].append((float(p), a))
    Z = np.load('/workspace/cl/ideas/r4_ilp/%s/%s.npz' % (st, name))
    inode = set(int(x) for x in Z['nodes'].tolist())
    iedge = set(zip((int(x) for x in Z['src'].tolist()), (int(x) for x in Z['dst'].tolist())))
    kept = set(nodes)
    drop_by_t = defaultdict(list)
    for i, t in zip(fids.tolist(), fT.tolist()):
        if int(i) not in kept: drop_by_t[int(t)].append(int(i))
    dtree = {t: (v, cKDTree(np.stack([fpos[fidx[x]] for x in v]))) for t, v in drop_by_t.items() if v}

    # ---------------- FN census (P15) ----------------
    fn = Counter()
    for s_, d_ in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()):
        s_, d_ = int(s_), int(d_); ps, pd_ = g2p.get(s_), g2p.get(d_)
        if ps is not None and pd_ is not None and pd_ in succ.get(ps, []): fn['tp'] += 1; continue
        if ps is None and pd_ is None: fn['both_unm'] += 1
        elif ps is None: fn['src_unm_dst_%s' % ('free' if not par.get(pd_) else 'taken')] += 1
        elif pd_ is None: fn['dst_unm_src_%s' % ('free' if not succ.get(ps) else 'taken')] += 1
        else:
            fs, fd = not succ.get(ps), not par.get(pd_)
            fn['both_m_%s_%s' % ('srcfree' if fs else 'srctaken', 'dstfree' if fd else 'dsttaken')] += 1
    # which unmatched GT nodes have a dropped detection within 7 um / a dropped ILP detection
    ucov = Counter()
    for t, (v, tr) in gut.items():
        for g in v:
            if t in dtree:
                ii = dtree[t][1].query_ball_point(gpos[g], 7.0)
                ids = [dtree[t][0][i] for i in ii]
                ucov['ilp' if any(x in inode for x in ids) else ('det' if ids else 'none')] += 1
            else: ucov['none'] += 1

    # ---------------- walks ----------------
    added_pos = defaultdict(list)
    used = set(); joined_start = set(); joined_end = set()

    def dupd(x):
        j = fidx[x]; t = int(fT[j]); d = 99.
        if t in ptree: d = float(ptree[t].query(fpos[j], k=1)[0])
        for q in added_pos.get(t, []): d = min(d, float(np.linalg.norm(q - fpos[j])))
        return d

    def gt_of_new(x):  # nearest P15-unmatched GT node within 7 um of detection x
        j = fidx[x]; t = int(fT[j])
        if t not in gut: return []
        v, tr = gut[t]
        return [v[i] for i in tr.query_ball_point(fpos[j], 7.0)]

    def label(g_src, cand_dst, fwd):
        """g_src: GT node of the existing (kept or added) side; cand_dst: GT nodes near the new node. returns lab, gt of new node"""
        nxt = (gsucc.get(g_src, []) if fwd else gpar.get(g_src, [])) if g_src is not None else []
        for c in cand_dst:
            if c in nxt: return 'TP', c
        gnew = cand_dst[0] if cand_dst else None
        valid = bool(nxt) or (gnew is not None and bool(gpar.get(gnew) if fwd else gsucc.get(gnew)))
        return ('FP' if valid else 'NE'), gnew

    def track_feats(n, fwd):
        # track length behind the terminus and last displacement
        L = 1; c = n; chain = [n]
        while L < 50:
            nx = par.get(c, []) if fwd else succ.get(c, [])
            if len(nx) != 1: break
            c = nx[0]; L += 1; chain.append(c)
        vel = ppos[chain[0]] - ppos[chain[1]] if len(chain) >= 2 else np.zeros(3)
        return L, vel, chain

    seeds = []
    for n, v in nodes.items():
        t = int(v['t'])
        if not succ.get(n) and t < T1: seeds.append((n, True))
        if not par.get(n) and t > T0: seeds.append((n, False))
    # order: by best first-step prob into a dropped detection (desc), so contested detections go to the strongest walk
    def first_p(sd):
        n, fwd = sd
        c = [p for p, x in (fch.get(n, []) if fwd else fpa.get(n, [])) if x not in kept]
        return max(c) if c else -1.
    seeds.sort(key=lambda sd: -first_p(sd))
    walks = []; tstat = Counter()
    for n, fwd in seeds:
        g0 = p2g.get(n)
        nxt0 = (gsucc.get(g0, []) if fwd else gpar.get(g0, [])) if g0 is not None else []
        if g0 is None: gs = 'unm'
        elif not nxt0: gs = 'stops'
        else:
            m = [g2p.get(x) for x in nxt0]
            if any(q is None for q in m): gs = 'cont_unm'
            elif any((not par.get(q)) if fwd else (not succ.get(q)) for q in m): gs = 'cont_free'
            else: gs = 'cont_taken'
        L, vel, chain = track_feats(n, fwd)
        t0 = int(nodes[n]['t'])
        cur = n; gcur = g0; steps = []; join = None; curpos = ppos[n]; curvel = vel
        for k in range(KMAX):
            t = int(nodes[cur]['t']) if cur in kept else int(fT[fidx[cur]])
            tn = t + 1 if fwd else t - 1
            if tn < T0 or tn > T1: break
            cands = sorted([(p, x) for p, x in (fch.get(cur, []) if fwd else fpa.get(cur, [])) if x not in kept and x not in used and p >= PW], reverse=True)
            pick = None; nalt = len(cands); p2 = cands[1][0] if len(cands) > 1 else 0.
            for p, x in cands:
                dd = dupd(x)
                if dd > DUP: pick = (p, x, dd); break
            if pick is None: break
            p, x, dd = pick
            j = fidx[x]
            cg = gt_of_new(x)
            lab, gnew = label(gcur, cg, fwd)
            pred_pos = curpos + curvel
            step = dict(x=x, t=tn, z=float(fV[j][0]), y=float(fV[j][1]), xx=float(fV[j][2]), p=round(p, 4), p2=round(p2, 4), nalt=nalt,
                        ilpn=int(x in inode), ilpe=int(((cur, x) if fwd else (x, cur)) in iedge), dup=round(dd, 2),
                        dist=round(float(np.linalg.norm(fpos[j] - curpos)), 2), vdev=round(float(np.linalg.norm(fpos[j] - pred_pos)), 2),
                        lab=lab, dens=int(len(ptree[tn].query_ball_point(fpos[j], 8.0))) if tn in ptree else 0)
            steps.append(step); used.add(x); added_pos[tn].append(fpos[j])
            curvel = fpos[j] - curpos; curpos = fpos[j]
            gcur = gnew
            cur = x
            # join check
            far = sorted([(pp, y) for pp, y in (fch.get(x, []) if fwd else fpa.get(x, [])) if y in kept and pp >= PW], reverse=True)
            for pp, y in far:
                ty = int(nodes[y]['t'])
                if ty != (tn + 1 if fwd else tn - 1): continue
                if fwd and (par.get(y) or y in joined_start): continue
                if (not fwd) and (succ.get(y) or y in joined_end): continue
                gy = p2g.get(y)
                if fwd: jl = 'TP' if (gcur is not None and gy is not None and gy in gsucc.get(gcur, [])) else (
                    'FP' if ((gcur is not None and gsucc.get(gcur)) or (gy is not None and gpar.get(gy))) else 'NE')
                else: jl = 'TP' if (gcur is not None and gy is not None and gy in gpar.get(gcur, [])) else (
                    'FP' if ((gcur is not None and gpar.get(gcur)) or (gy is not None and gsucc.get(gy))) else 'NE')
                join = dict(y=y, p=round(pp, 4), lab=jl, dist=round(float(np.linalg.norm(ppos[y] - curpos)), 2))
                (joined_start if fwd else joined_end).add(y)
                break
            if join is not None: break
        near = None
        if not steps:
            tn = t0 + 1 if fwd else t0 - 1
            if T0 <= tn <= T1 and tn in dtree:
                q = ppos[n] + vel
                dd_, ii = dtree[tn][1].query(q, k=1)
                if dd_ <= RN:
                    x = dtree[tn][0][int(ii)]
                    if x not in used and dupd(x) > DUP:
                        cg = gt_of_new(x); lab, _ = label(g0, cg, fwd)
                        j = fidx[x]
                        pe = [pp for pp, xx in (fch.get(n, []) if fwd else fpa.get(n, [])) if xx == x]
                        near = dict(x=x, t=tn, z=float(fV[j][0]), y=float(fV[j][1]), xx=float(fV[j][2]), dist=round(float(dd_), 2),
                                    raw=round(float(np.linalg.norm(fpos[j] - ppos[n])), 2), dup=round(dupd(x), 2), ilpn=int(x in inode), lab=lab,
                                    pedge=int(bool(pe)), p=round(max(pe), 4) if pe else 0.0,
                                    dens=int(len(ptree[tn].query_ball_point(fpos[j], 8.0))) if tn in ptree else 0)
        tstat[(gs, 'has_step' if steps else ('near' if near else 'none'))] += 1
        if steps or near:
            walks.append(dict(seed=n, fwd=int(fwd), t0=t0, L=L, gs=gs, steps=steps, join=join, near=near,
                              vlen=round(float(np.linalg.norm(vel)), 2), T0=T0, T1=T1))
    os.makedirs('%s/%s' % (OUT, st), exist_ok=True)
    json.dump(dict(movie=name, set=st, walks=walks), open('%s/%s/%s.json' % (OUT, st, name), 'w'))
    return dict(movie=name, set=st, emb=name[:4], fn=dict(fn), ucov=dict(ucov), tstat={'|'.join(k): v for k, v in tstat.items()},
                n_det=len(fids), n_nodes=len(nodes), n_total=n_total)


if __name__ == '__main__':
    files = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s))]
    if len(sys.argv) > 1: files = files[:int(sys.argv[1])]
    with Pool(int(os.environ.get('RULE_POOL', '24')), maxtasksperchild=4) as p: R = p.map(job, files, chunksize=1)
    json.dump(R, open('/workspace/cl/nm/ver_census.json', 'w'))
    print('done', len(R))
