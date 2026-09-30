"""dv_cand (feature table for the division-completion feasibility probe; reads GT for labels only).
On P15 graphs: all div_complete candidates (p,a,b,q,typ) with labels
  P = would recover a not-yet-recovered GT division, D = GT division already recovered (adding = FP), X = cross-component fork (FP),
  N = p is an annotated non-terminal GT cell and not a division (FP), U = not evaluable (kept as 1% sample).
Plus synthetic 'birth' rows from every existing P15 fork F->(c1,c2): the edge F->c_i is removed and (F, c_other, c_i) is described as a
start-type candidate (weak positives; label = official fork label tp/fp/nc).
Features: geometry/structure + cheap image intensity statistics of p (t, t-1, t-2, history), a, b (t+1, t+2) and q.
usage: dv_cand.py -> /workspace/cl/nm/dv_cand/<set>__<movie>.json"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
ART = '/workspace/art_b56/artifact_bundle'; os.environ['BIOHUB_ART'] = ART
for p in ['/workspace/cl', '/workspace/official/src', '/workspace/code', ART, '/workspace/p12ds']:
    sys.path.insert(0, p)
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
OUT = Path('/workspace/cl/nm/dv_cand')


def img_stats(name, nodes):
    """per node: core mean (z+-1,y/x+-4px), shell mean (z+-3, y/x+-12px), core max; each divided by the frame median over all nodes."""
    import numcodecs.blosc
    numcodecs.blosc.use_threads = False
    import zarr
    arr = zarr.open_group('/workspace/data/train/%s.zarr' % name, mode='r')['0']
    T, Z, Y, X = arr.shape
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    res = {}
    for t, ns in byt.items():
        fr = np.asarray(arr[t], np.float32)
        c = np.array([[nodes[n]['z'], nodes[n]['y'], nodes[n]['x']] for n in ns])
        c = np.rint(c).astype(int)
        c[:, 0] = np.clip(c[:, 0], 0, Z - 1); c[:, 1] = np.clip(c[:, 1], 0, Y - 1); c[:, 2] = np.clip(c[:, 2], 0, X - 1)
        vals = np.zeros((len(ns), 3), np.float32)
        for i, (z, y, x) in enumerate(c):
            core = fr[max(0, z - 1):z + 2, max(0, y - 4):y + 5, max(0, x - 4):x + 5]
            sh = fr[max(0, z - 3):z + 4, max(0, y - 12):y + 13, max(0, x - 12):x + 13]
            vals[i] = (core.mean(), sh.mean(), core.max())
        med = np.median(vals, 0) + 1e-3
        for n, v in zip(ns, vals / med): res[n] = v
    return res


def job(a):
    s, name = a
    of = OUT / ('%s__%s.json' % (s, name))
    if of.exists(): return 1
    import evalx, div_complete as dc
    from tracking_cellmot.division_metrics import score_divisions, _match_full, _matched_node_attrs
    from scipy.spatial import cKDTree
    K = evalx.K
    rng = np.random.default_rng(abs(hash(name)) % 2 ** 31)
    nodes, edges = evalx.load_graph_json('/workspace/cl/ps_p15_%s/graphs/%s.json' % (s, name))
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    recovered = {int(d) for d, v in res.scores.items() if v}
    tpf = {inv[int(x)] for x in res.tp_forks}; fpf = {inv[int(x)] for x in res.fp_forks}
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(x)]: int(y) for x, y in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    ea = gt.edge_attrs(); gs = defaultdict(list); gp = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(x)].append(int(y)); gp[int(y)] = int(x)
    comp = {}
    for g0 in gt.node_ids():
        g0 = int(g0)
        if g0 in comp: continue
        comp[g0] = g0; st = [g0]
        while st:
            c = st.pop()
            for nb in gs.get(c, []) + ([gp[c]] if c in gp else []):
                if nb not in comp: comp[nb] = g0; st.append(nb)
    rows, out, prev, pos = dc.candidates(nodes, edges)
    tt = {n: int(v['t']) for n, v in nodes.items()}
    I = img_stats(name, nodes)
    frames = defaultdict(list)
    for n in nodes: frames[tt[n]].append(n)
    trees = {t: cKDTree(np.array([pos[n] for n in ns])) for t, ns in frames.items()}
    ends = defaultdict(list)
    for n in nodes:
        if not out.get(n): ends[tt[n]].append(n)
    etrees = {t: cKDTree(np.array([pos[n] for n in ns])) for t, ns in ends.items()}
    starts = defaultdict(list)
    for n in nodes:
        if n not in prev and tt[n] > 0: starts[tt[n]].append(n)
    strees = {t: cKDTree(np.array([pos[n] for n in ns])) for t, ns in starts.items()}
    forks = {n for n in nodes if len(out.get(n, [])) >= 2}

    def lab_of(p, a_, b_):
        g_p, g_a, g_b = p2g.get(p), p2g.get(a_), p2g.get(b_)
        if g_a is not None and g_b is not None and comp.get(g_a) != comp.get(g_b): return 'X'
        if g_p is None or not gs.get(g_p): return 'U'
        for gd in [g_p] + ([gp[g_p]] if g_p in gp else []) + list(gs.get(g_p, [])):
            chs = gs.get(gd, [])
            if len(chs) != 2: continue
            lin = [set([x] + gs.get(x, [])) for x in chs]
            if g_a is not None and g_b is not None and any(g_a in L for L in lin) and any(g_b in L for L in lin) and not any(g_a in L and g_b in L for L in lin):
                return ('D' if gd in recovered else 'P') + ':%d' % gd
        return 'N'

    def feats(p, a_, b_, q, typ, out, prev):
        def back(n, k=40):
            c = [n]
            while len(c) < k and c[-1] in prev and len(out.get(prev[c[-1]], [])) == 1: c.append(prev[c[-1]])
            return c

        def fwd(n, k=40):
            c = [n]
            while len(c) < k and len(out.get(c[-1], [])) == 1: c.append(out[c[-1]][0])
            return c
        t = tt[p]
        ph = back(p); af = fwd(a_); bf = fwd(b_)
        vp = (pos[ph[0]] - pos[ph[min(3, len(ph) - 1)]]) / max(1, min(3, len(ph) - 1))
        f = dict(typ=int(typ == 'stolen'), t=t, z=float(pos[p][0]), border=float(min(pos[p][1], pos[p][2], 256 * .40625 - pos[p][1], 256 * .40625 - pos[p][2])),
                 p_hist=len(ph), a_fut=len(af), b_fut=len(bf), p_root=int(ph[-1] not in prev), p_after_div=int(ph[-1] in prev))
        va, vb = pos[a_] - pos[p], pos[b_] - pos[p]
        f['d_pb'] = float(np.linalg.norm(vb)); f['d_ab'] = float(np.linalg.norm(pos[a_] - pos[b_])); f['d_pa'] = float(np.linalg.norm(va))
        f['cos_ab'] = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-6))
        f['mid_dev'] = float(np.linalg.norm((pos[a_] + pos[b_]) / 2 - pos[p] - vp)); f['vel_p'] = float(np.linalg.norm(vp))
        for k in [2, 4, 8]:
            f['d_ab_%d' % k] = float(np.linalg.norm(pos[af[min(k, len(af) - 1)]] - pos[bf[min(k, len(bf) - 1)]]))
        # frames since the p-track was born by a fork (40 = none within 40); frames until a or b track forks
        f['p_div_age'] = len(ph) if ph[-1] in prev else 40
        f['ab_next_fork'] = min([i for i, n in enumerate(af) if n in forks] + [i for i, n in enumerate(bf) if n in forks] + [40])
        f['dens_p'] = len(trees[t].query_ball_point(pos[p], 8.0)); f['dens_b'] = len(trees[t + 1].query_ball_point(pos[b_], 8.0))
        # alternative explanations for b being a start: track ends at t (or t-1) near b
        for dt in [0, 1]:
            et = etrees.get(t - dt)
            if et is None: f['end%d_d' % dt] = 30.; f['end%d_n' % dt] = 0; continue
            dd, _ = et.query(pos[b_]); f['end%d_d' % dt] = float(min(dd, 30.)); f['end%d_n' % dt] = len(et.query_ball_point(pos[b_], 10.0))
        stt = strees.get(t + 1)
        f['starts_near_p'] = len(stt.query_ball_point(pos[p], 13.0)) if stt is not None else 0
        if q is not None:
            qh = back(q)
            f['q_hist'] = len(qh); f['q_root'] = int(qh[-1] not in prev); f['d_qp'] = float(np.linalg.norm(pos[q] - pos[p])); f['d_qb'] = float(np.linalg.norm(pos[q] - pos[b_]))
            tq = tt[qh[-1]]; pn = [x for x in ph if tt[x] == tq]
            f['qstart_dp'] = float(np.linalg.norm(pos[qh[-1]] - pos[pn[0]])) if pn else -1.
            dd = [float(np.linalg.norm(pos[x] - pos[y])) for x in qh[:10] for y in ph[:10] if tt[x] == tt[y]]
            f['qp_min'] = min(dd) if dd else -1.; f['qp_mean'] = float(np.mean(dd)) if dd else -1.
            f['Iq'] = float(I[q][0]); f['Cq'] = float(I[q][0] / (I[q][1] + 1e-3))
        # image
        def ist(n): return I[n]
        f['Ip0'], f['Sp0'], f['Mp0'] = map(float, ist(ph[0]))
        f['Ip1'] = float(ist(ph[min(1, len(ph) - 1)])[0]); f['Ip2'] = float(ist(ph[min(2, len(ph) - 1)])[0])
        hist = [ist(n)[0] for n in ph[4:12]]
        f['Iph'] = float(np.mean(hist)) if hist else -1.
        f['Ia'], f['Sa'], f['Ma'] = map(float, ist(a_)); f['Ib'], f['Sb'], f['Mb'] = map(float, ist(b_))
        f['Ia2'] = float(ist(af[min(2, len(af) - 1)])[0]); f['Ib2'] = float(ist(bf[min(2, len(bf) - 1)])[0])
        f['Ia5'] = float(ist(af[min(5, len(af) - 1)])[0]); f['Ib5'] = float(ist(bf[min(5, len(bf) - 1)])[0])
        f['r_ab'] = min(f['Ia'], f['Ib']) / (max(f['Ia'], f['Ib']) + 1e-3)
        f['r_abp'] = (f['Ia'] + f['Ib']) / 2 / (f['Ip0'] + 1e-3)
        f['r_ph'] = f['Ip0'] / (f['Iph'] + 1e-3) if f['Iph'] > 0 else -1.
        f['Cp'] = f['Ip0'] / (f['Sp0'] + 1e-3); f['Ca'] = f['Ia'] / (f['Sa'] + 1e-3); f['Cb'] = f['Ib'] / (f['Sb'] + 1e-3)
        return f
    R = []
    ncp = defaultdict(int); ncb = defaultdict(int)
    for p, a_, b_, q, typ in rows: ncp[p] += 1; ncb[b_] += 1
    for p, a_, b_, q, typ in rows:
        lab = lab_of(p, a_, b_)
        if lab == 'U' and rng.random() > 0.01: continue
        f = feats(p, a_, b_, q, typ, out, prev)
        f.update(ncand_p=ncp[p], ncand_b=ncb[b_], lab=lab, src='cand', p=p, a=a_, b=b_, q=q)
        R.append(f)
    # synthetic births from existing forks
    for F in forks:
        ch = out[F]
        if len(ch) != 2: continue
        flab = 'tp' if F in tpf else ('fp' if F in fpf else 'nc')
        for i in range(2):
            b_, a_ = ch[i], ch[1 - i]
            o2 = out.copy(); o2[F] = [a_]
            p2 = dict(prev); del p2[b_]
            f = feats(F, a_, b_, None, 'start', o2, p2)
            f.update(ncand_p=ncp.get(F, 0), ncand_b=ncb.get(b_, 0), lab='F' + flab, src='fork', p=F, a=a_, b=b_, q=None)
            R.append(f)
    of.write_text(json.dumps(R))
    return len(R)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, Path(f).stem) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s))]
    if len(sys.argv) > 1: jobs = jobs[:int(sys.argv[1])]
    with Pool(24, maxtasksperchild=2) as p: R = p.map(job, jobs, chunksize=1)
    print('done', len(R), sum(R))
