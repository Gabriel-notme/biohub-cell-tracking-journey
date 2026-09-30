"""General link re-scoring candidates on P15 graphs (all 199 movies).
Candidate (s, d): s at t with <= 1 child, d at t+1 within R um, d not a child of s, d not a fork daughter.
Action: add s->d, drop s->cur_d and cur_s->d.  Optional swap completion cur_s->cur_d (recorded, labelled, featurised).
Geometric / motion (+-3 frames) / competitor / pre-ILP / B5 edge-prob features.  Only EVALUABLE rows are stored with
features (any touched edge touches an annotated GT track); the total candidate count is recorded for runtime estimates.
-> /workspace/cl/nm/rl_cands/<set>__<movie>.pkl"""
import os, sys, json, glob, pickle
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])
R = 14.0
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
OUT = Path('/workspace/cl/nm/rl_cands'); OUT.mkdir(exist_ok=True, parents=True)
FEATS = ['typ', 'dist', 'dz', 'dxy', 'd_sc', 'd_qd', 'd_qc', 'fe_sd', 'fe_sc', 'fe_qd', 'fe_qc', 'ep_sc', 'ep_qd', 'lk_sc', 'lk_qd',
         'hist_s', 'fut_d', 'fut_c', 'hist_q', 'sp_s', 'sp_d', 'err_f1', 'err_f3', 'err_b1', 'err_b3', 'err_fc', 'err_bq',
         'rms_new', 'rms_sc', 'rms_qd', 'rms_qc', 'acc_new', 'acc_sc', 'acc_qd', 'cos_new', 'cos_sc',
         'n_s', 'rk_s', 'rkp_s', 'n_d', 'rk_d', 'rkp_d', 'gap_s', 'gap_d', 'dens_s', 'dens_d', 'z', 'tt', 'sdaught', 'fut_c_div', 'c_alt', 'q_alt',
         'nodes_t', 'ends_t', 'starts_t1']
LINKATTR = ('relink', 'edge_link', 'long_link', 'tb_ext', 'tb_join', 'div_complete', 'dup_join', 'gap_closed', 'gap2_recovered')


def linefit_rms(P):
    """RMS residual of a straight constant-velocity fit to consecutive positions P (k x 3)."""
    if len(P) < 3: return -1.
    t = np.arange(len(P), dtype=float); A = np.column_stack([t, np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, P, rcond=None)
    return float(np.sqrt(((A @ coef - P) ** 2).sum(1).mean()))


def accel(P, j):
    """|second difference| at the junction index j (between P[j] and P[j+1])."""
    if j < 1 or j + 1 >= len(P): return -1.
    return float(np.linalg.norm(P[j + 1] - 2 * P[j] + P[j - 1]))


def job(a):
    s_, f = a
    name = Path(f).stem; of = OUT / ('%s__%s.pkl' % (s_, name))
    if of.exists(): return 1
    import numcodecs.blosc
    numcodecs.blosc.use_threads = False
    import evalx, edge_link
    from loeo import gt_maps
    from scipy.spatial import cKDTree
    nodes, edges = evalx.load_graph_json(f)
    fids, fT, fV, fE, fprob = edge_link.load_full(FULL[s_] + '/' + name + '.geff')
    fedge = {(int(x), int(y)): float(p) for (x, y), p in zip(fE.tolist(), fprob.tolist())}
    p2g, gs, gp = gt_maps(name, nodes, edges)
    succ = defaultdict(list); par = {}; eat = {}
    for e in edges:
        x, y = int(e['source_id']), int(e['target_id']); succ[x].append(y); par[y] = x; eat[(x, y)] = e
    pos = {n: np.array([v['z'], v['y'], v['x']], float) * S for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    trees = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in frames.items()}
    T = max(frames)
    ends_t = Counter(int(nodes[n]['t']) for n in nodes if not succ.get(n)); starts_t = Counter(int(nodes[n]['t']) for n in nodes if n not in par)
    forkpar = {x for x, c in succ.items() if len(c) >= 2}; daught = {y for x in forkpar for y in succ[x]}

    def valid(x, y):
        gx, gy = p2g.get(x), p2g.get(y)
        return (gx is not None and len(gs.get(gx, [])) > 0) or (gy is not None and gy in gp)

    def tp(x, y):
        gx, gy = p2g.get(x), p2g.get(y)
        return gx is not None and gy is not None and gy in gs.get(gx, [])

    def hist(n, k=3):
        P = [pos[n]]; c = n
        while len(P) <= k and c in par: c = par[c]; P.append(pos[c])
        return P[::-1]  # oldest .. n

    def fut(n, k=3):
        P = [pos[n]]; c = n
        while len(P) <= k and len(succ.get(c, [])) == 1: c = succ[c][0]; P.append(pos[c])
        return P  # n .. newest

    def hlen(n, lim=40):
        k = 0
        while n in par and k < lim: n = par[n]; k += 1
        return k

    def flen(n, lim=40):
        k = 0
        while len(succ.get(n, [])) == 1 and k < lim: n = succ[n][0]; k += 1
        return k

    def vel(P):  # mean velocity of an ordered trajectory
        return (P[-1] - P[0]) / (len(P) - 1) if len(P) > 1 else None
    ntot = 0; rows = []
    cache_h, cache_f = {}, {}
    H = lambda n: cache_h.setdefault(n, hist(n)); F = lambda n: cache_f.setdefault(n, fut(n))
    # predecessor candidate lists for competitor features
    for t in sorted(frames):
        if t + 1 not in trees: continue
        ns1, tr1 = trees[t + 1]
        src = [n for n in frames[t] if len(succ.get(n, [])) <= 1]
        if not src: continue
        nb = tr1.query_ball_point(np.array([pos[n] for n in src]), R)
        back = defaultdict(list)
        for sn, js in zip(src, nb):
            for j in js: back[ns1[j]].append(sn)
        for sn, js in zip(src, nb):
            cands = [ns1[j] for j in js if ns1[j] not in succ.get(sn, []) and ns1[j] not in daught]
            if not cands: continue
            Ps = H(sn); vs1 = (Ps[-1] - Ps[-2]) if len(Ps) > 1 else None; vs = vel(Ps)
            dl = {dn: float(np.linalg.norm(pos[dn] - pos[sn])) for dn in [ns1[j] for j in js]}
            pe = {dn: float(np.linalg.norm(pos[dn] - (pos[sn] + (vs if vs is not None else 0)))) for dn in dl}
            sd_ = sorted(dl.values()); sp_ = sorted(pe.values())
            cd = succ.get(sn, [None])[0] if succ.get(sn) else None
            for dn in cands:
                ntot += 1
                q = par.get(dn)
                ev = valid(sn, dn) or (cd is not None and valid(sn, cd)) or (q is not None and valid(q, dn))
                if not ev: continue
                typ = (0 if (cd is None and q is None) else 1 if cd is None else 2 if q is None else 3)
                Pd = F(dn); vd = vel(Pd); vd1 = (Pd[1] - Pd[0]) if len(Pd) > 1 else None
                disp = pos[dn] - pos[sn]; dist = dl[dn]
                r = dict(s=sn, d=dn, c=cd, q=q, t=t, typ=typ, dist=dist, dz=float(abs(disp[0])), dxy=float(np.linalg.norm(disp[1:])),
                         d_sc=float(np.linalg.norm(pos[cd] - pos[sn])) if cd is not None else -1.,
                         d_qd=float(np.linalg.norm(pos[dn] - pos[q])) if q is not None else -1.,
                         d_qc=float(np.linalg.norm(pos[cd] - pos[q])) if (q is not None and cd is not None) else -1.,
                         fe_sd=fedge.get((sn, dn), -1.), fe_sc=fedge.get((sn, cd), -1.) if cd is not None else -2.,
                         fe_qd=fedge.get((q, dn), -1.) if q is not None else -2., fe_qc=fedge.get((q, cd), -1.) if (q is not None and cd is not None) else -2.)
                esc = eat.get((sn, cd), {}) if cd is not None else {}; eqd = eat.get((q, dn), {}) if q is not None else {}
                r['ep_sc'] = float(esc.get('edge_prob', -1.) or -1.) if cd is not None else -2.; r['ep_qd'] = float(eqd.get('edge_prob', -1.) or -1.) if q is not None else -2.
                r['lk_sc'] = (1 + next((i for i, k in enumerate(LINKATTR) if k in esc), -1)) if cd is not None else -1
                r['lk_qd'] = (1 + next((i for i, k in enumerate(LINKATTR) if k in eqd), -1)) if q is not None else -1
                r['hist_s'] = hlen(sn); r['fut_d'] = flen(dn); r['fut_c'] = flen(cd) if cd is not None else -1; r['hist_q'] = hlen(q) if q is not None else -1
                r['sp_s'] = float(np.linalg.norm(vs)) if vs is not None else -1.; r['sp_d'] = float(np.linalg.norm(vd)) if vd is not None else -1.
                r['err_f1'] = float(np.linalg.norm(pos[dn] - pos[sn] - vs1)) if vs1 is not None else -1.
                r['err_f3'] = pe[dn] if vs is not None else -1.
                r['err_b1'] = float(np.linalg.norm(pos[sn] - (pos[dn] - vd1))) if vd1 is not None else -1.
                r['err_b3'] = float(np.linalg.norm(pos[sn] - (pos[dn] - vd))) if vd is not None else -1.
                r['err_fc'] = float(np.linalg.norm(pos[cd] - pos[sn] - vs)) if (cd is not None and vs is not None) else -1.
                if q is not None and vd is not None: r['err_bq'] = float(np.linalg.norm(pos[q] - (pos[dn] - vd)))
                else: r['err_bq'] = -1.
                Tn = np.array(Ps + Pd); r['rms_new'] = linefit_rms(Tn); r['acc_new'] = accel(Tn, len(Ps) - 1)
                if cd is not None:
                    Tc = np.array(Ps + F(cd)); r['rms_sc'] = linefit_rms(Tc); r['acc_sc'] = accel(Tc, len(Ps) - 1)
                    r['cos_sc'] = float(vs @ (pos[cd] - pos[sn]) / (np.linalg.norm(vs) * r['d_sc'] + 1e-6)) if vs is not None else 0.
                else: r['rms_sc'] = r['acc_sc'] = -1.; r['cos_sc'] = 0.
                if q is not None:
                    Pq = H(q); Tq = np.array(Pq + Pd); r['rms_qd'] = linefit_rms(Tq); r['acc_qd'] = accel(Tq, len(Pq) - 1)
                    r['rms_qc'] = linefit_rms(np.array(Pq + F(cd))) if cd is not None else -1.
                else: r['rms_qd'] = r['acc_qd'] = r['rms_qc'] = -1.
                r['cos_new'] = float(vs @ disp / (np.linalg.norm(vs) * dist + 1e-6)) if vs is not None else 0.
                r['n_s'] = len(dl); r['rk_s'] = sd_.index(dist); r['rkp_s'] = sp_.index(pe[dn])
                bl = back.get(dn, []); dd = sorted(float(np.linalg.norm(pos[dn] - pos[x])) for x in bl)
                r['n_d'] = len(bl); r['rk_d'] = dd.index(dist) if dist in dd else -1
                if vd is not None:
                    bp = sorted(float(np.linalg.norm(pos[x] - (pos[dn] - vd))) for x in bl); r['rkp_d'] = bp.index(r['err_b3']) if r['err_b3'] in bp else -1
                else: r['rkp_d'] = -1
                r['gap_s'] = (sd_[1] - dist) if (r['rk_s'] == 0 and len(sd_) > 1) else (dist - sd_[0])
                r['gap_d'] = (dd[1] - dist) if (r['rk_d'] == 0 and len(dd) > 1) else ((dist - dd[0]) if dd else -1.)
                r['dens_s'] = len(trees[t][1].query_ball_point(pos[sn], 8.0)); r['dens_d'] = len(tr1.query_ball_point(pos[dn], 8.0))
                r['z'] = float(pos[sn][0]); r['tt'] = t / T; r['sdaught'] = int(sn in daught)
                r['fut_c_div'] = int(cd is not None and cd in forkpar)
                r['c_alt'] = len(back.get(cd, [])) if cd is not None else -1
                r['q_alt'] = int(q is not None and len(succ.get(q, [])) == 1)
                r['nodes_t'] = len(frames[t]); r['ends_t'] = ends_t[t]; r['starts_t1'] = starts_t[t + 1]
                # labels
                r['tp_sd'] = int(tp(sn, dn)); r['v_sd'] = int(valid(sn, dn))
                r['tp_sc'] = int(cd is not None and tp(sn, cd)); r['v_sc'] = int(cd is not None and valid(sn, cd))
                r['tp_qd'] = int(q is not None and tp(q, dn)); r['v_qd'] = int(q is not None and valid(q, dn))
                r['tp_qc'] = int(q is not None and cd is not None and tp(q, cd)); r['v_qc'] = int(q is not None and cd is not None and valid(q, cd))
                rows.append(r)
    X = np.array([[r[k] for k in FEATS] for r in rows], np.float32).reshape(-1, len(FEATS))
    L = {k: np.array([r[k] for r in rows]) for k in ['s', 'd', 't', 'tp_sd', 'v_sd', 'tp_sc', 'v_sc', 'tp_qd', 'v_qd', 'tp_qc', 'v_qc']}
    L['c'] = np.array([r['c'] if r['c'] is not None else -1 for r in rows]); L['q'] = np.array([r['q'] if r['q'] is not None else -1 for r in rows])
    pickle.dump(dict(X=X, L=L, ntot=ntot, nnodes=len(nodes)), open(of, 'wb'))
    return 1


if __name__ == '__main__':
    jobs = [(s, f) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s))]
    if len(sys.argv) > 1: jobs = jobs[:int(sys.argv[1])]
    with Pool(int(os.environ.get('NPROC', '20')), maxtasksperchild=4) as p: print('done', sum(p.map(job, jobs, chunksize=1)), len(jobs))
