"""r2_caps (diagnostic, reads GT): blind-spot lens on P14 (= P13 graphs + ideas.combo14).
For every GT edge / GT division, which structural cap of the P-stage / B5 output blocks it; and for every pre-ILP candidate edge that is
absent from P14, its structural type + label (TP / evaluable FP / unevaluable) and the label of the edge it would displace.
Writes /workspace/cl/ideas/r2_caps_out.json"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
for p in ['/workspace/cl', '/workspace/cl/ideas', '/workspace/official/src', '/workspace/code', '/workspace/p56stage']:
    sys.path.insert(0, p)
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}


def febin(p):
    if p is None: return 'none'
    return '>=.9' if p >= .9 else ('.5-.9' if p >= .5 else ('.2-.5' if p >= .2 else '<.2'))


def dbin(d):
    return '<=6' if d <= 6 else ('6-10' if d <= 10 else ('10-14' if d <= 14 else ('14-18' if d <= 18 else '>18')))


def job(f):
    try:
        import numcodecs.blosc
        numcodecs.blosc.use_threads = False
    except Exception:
        pass
    import evalx, edge_link, combo14
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs, score_divisions
    K = evalx.K
    name = Path(f).stem; st = f.split('ps_p13_')[1].split('/')[0]; emb = name[:4]
    fg = FULL[st] + '/' + name + '.geff'
    nodes, edges = evalx.load_graph_json(f)
    nodes, edges, _ = combo14.apply(nodes, edges, fullgeff=fg)
    fids, fT, fV, fE, fprob = edge_link.load_full(fg)
    fedge = {(int(a), int(b)): float(p) for (a, b), p in zip(fE.tolist(), fprob.tolist())}
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    g2p = {v: k for k, v in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x]) * S for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    gt_t = {int(i): int(t) for i, t in zip(ga[K.NODE_ID].to_list(), ga['t'].to_list())}
    ea = gt.edge_attrs()
    gE = [(int(x), int(y)) for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list())]
    gEs = set(gE)
    gout = defaultdict(list); gin = {}
    for s, d in gE: gout[s].append(d); gin[d] = s
    out = defaultdict(list); par = {}; eflag = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
        eflag[(a, b)] = ('ll' if 'long_link' in e else 'el' if 'edge_link' in e else 'rl' if 'relink' in e else 'dc' if 'div_complete' in e else
                         'dj' if 'dup_join' in e else 'gc' if 'gap_closed' in e else 'g2' if 'gap2_recovered' in e else 'mr' if 'motion_relinked' in e else 'oth')
    ppos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx'], float) * S for n, v in nodes.items()}

    def pdist(a, b): return float(np.linalg.norm(ppos[a] - ppos[b]))

    def lab(a, b):
        ga_, gb_ = p2g.get(a), p2g.get(b)
        if ga_ is not None and gb_ is not None and (ga_, gb_) in gEs: return 'TP'
        if (ga_ is not None and gout.get(ga_)) or (gb_ is not None and gb_ in gin): return 'FPe'
        return 'U'

    def fwdlen(n, lim=30):
        k = 0
        while len(out.get(n, [])) == 1 and k < lim: n = out[n][0]; k += 1
        return k

    def backlen(n, lim=30):
        k = 0
        while n in par and k < lim: n = par[n]; k += 1
        return k

    def stype(a):  # source-side type
        k = len(out.get(a, []))
        return 'E' if k == 0 else ('F' if k >= 2 else '1')

    def dtype(b):  # target-side type
        q = par.get(b)
        if q is None: return 'S'
        return 'TF' if len(out[q]) >= 2 else 'T1'
    # dropped pre-ILP detections (for unmatched-endpoint FN attribution)
    fP = fV * S
    drop_by_t = defaultdict(list)
    for i, (fi, t) in enumerate(zip(fids.tolist(), fT.tolist())):
        if int(fi) not in nodes: drop_by_t[int(t)].append(i)
    from scipy.spatial import cKDTree
    dtree = {t: (ix, cKDTree(fP[ix])) for t, ix in drop_by_t.items() if ix}

    def dropped_near(g):
        t = gt_t[g]
        if t not in dtree: return False
        ix, tr = dtree[t]
        return bool(tr.query_ball_point(gpos[g], 7.0))
    ec = Counter(); rows = []
    # ---- GT edges
    for gs, gd in gE:
        ps, pdn = g2p.get(gs), g2p.get(gd)
        L = float(np.linalg.norm(gpos[gs] - gpos[gd]))
        if ps is None or pdn is None:
            miss = 'miss_both' if (ps is None and pdn is None) else ('miss_s' if ps is None else 'miss_d')
            dn = any(dropped_near(g) for g, p in ((gs, ps), (gd, pdn)) if p is None)
            ec[(emb, 'FN', miss + ('_dropdet' if dn else '_nodet'), '', dbin(L))] += 1
            continue
        if pdn in out.get(ps, []):
            ec[(emb, 'TP', eflag[(ps, pdn)], '', dbin(L))] += 1; continue
        flip = any(c != pdn and float(np.linalg.norm(ppos[c] - gpos[gd])) <= 7 for c in out.get(ps, [])) or \
            (par.get(pdn) is not None and par[pdn] != ps and float(np.linalg.norm(ppos[par[pdn]] - gpos[gs])) <= 7)
        cat = stype(ps) + '->' + dtype(pdn)
        fe = fedge.get((ps, pdn))
        d = pdist(ps, pdn)
        ec[(emb, 'FN', ('flip:' if flip else '') + cat, febin(fe), dbin(d))] += 1
        if not flip:
            r = dict(m=name, emb=emb, t=int(nodes[ps]['t']), cat=cat, fe=fe, d=d, L=L, gdiv=len(gout[gs]) >= 2, gdd=len(gout.get(gd, [])),
                     s_fresh=(ps in par and len(out[par[ps]]) >= 2), hist_s=backlen(ps), fut_d=fwdlen(pdn))
            if cat[0] == '1':
                c = out[ps][0]; r['cur_d_div'] = len(out.get(c, [])) >= 2; r['rm_s'] = lab(ps, c); r['rm_s_flag'] = eflag[(ps, c)]
            if cat.endswith('T1') or cat.endswith('TF'):
                q = par[pdn]; r['rm_d'] = lab(q, pdn); r['rm_d_flag'] = eflag[(q, pdn)]; r['q_fork'] = len(out[q]) >= 2
            rows.append(r)
    # ---- pre-ILP candidate universe (absent from P14)
    cu = Counter()
    for (a, b), p in fedge.items():
        if a not in nodes or b not in nodes or b in out.get(a, []): continue
        if int(nodes[b]['t']) != int(nodes[a]['t']) + 1: continue
        cat = stype(a) + '->' + dtype(b)
        rm = []
        if cat[0] == '1': rm.append(lab(a, out[a][0]))
        if cat.endswith('T1') or cat.endswith('TF'): rm.append(lab(par[b], b))
        cu[(emb, cat, febin(p), dbin(pdist(a, b)), lab(a, b), ','.join(rm))] += 1
    # ---- GT divisions
    res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    drows = []
    for gp, sc in res.scores.items():
        gp = int(gp); ch = gout.get(gp, [])
        r = dict(m=name, emb=emb, t=gt_t[gp], tp=int(sc))
        if sc: drows.append(r); continue
        pp = g2p.get(gp); pc = [g2p.get(c) for c in ch]
        r['pp'] = pp is not None; r['nch_m'] = sum(x is not None for x in pc)
        if pp is None:
            r['cls'] = 'miss_parent'
        elif None in pc:
            r['cls'] = 'miss_child'
            # which structure: parent out-degree, the matched child connected?
            r['p_out'] = len(out.get(pp, []))
            r['drop_child'] = any(dropped_near(c) for c, x in zip(ch, pc) if x is None)
        else:
            k = len(out.get(pp, []))
            pa, pb = pc
            r['p_out'] = k
            r['d_ab'] = pdist(pa, pb); r['d_pa'] = pdist(pp, pa); r['d_pb'] = pdist(pp, pb)
            r['fe_pa'] = fedge.get((pp, pa)); r['fe_pb'] = fedge.get((pp, pb))
            r['ta'] = dtype(pa); r['tb'] = dtype(pb)
            r['fa'] = fwdlen(pa); r['fb'] = fwdlen(pb)
            if k == 0: r['cls'] = 'p_end'
            elif k >= 2: r['cls'] = 'p_fork'
            else:
                c = out[pp][0]
                if c in (pa, pb):
                    b = pb if c == pa else pa
                    q = par.get(b)
                    if q is None: typ = 'start'
                    elif len(out[q]) >= 2: typ = 'b_forkdaughter'
                    elif q == pp: typ = 'weird'
                    else: typ = 'stolen'
                    caps = []
                    if pdist(pp, b) > 13: caps.append('pb>13')
                    if pdist(c, b) > 20: caps.append('ab>20')
                    if fwdlen(c) + 1 < 2 or fwdlen(b) + 1 < 2: caps.append('branch<2')
                    r['cls'] = 'p1_' + typ + ('[' + ','.join(caps) + ']' if caps else '')
                else:
                    r['cls'] = 'p1_other_child'
                    r['c_near'] = min(float(np.linalg.norm(ppos[c] - gpos[x])) for x in ch)
        drows.append(r)
    fork_c = Counter()
    for n in nodes:
        if len(out.get(n, [])) >= 2:
            fork_c[(emb, 'tp' if n in {inv[int(x)] for x in res.tp_forks} else ('fp' if n in {inv[int(x)] for x in res.fp_forks} else 'nc'))] += 1
    return dict(m=name, ec=[list(k) + [v] for k, v in ec.items()], cu=[list(k) + [v] for k, v in cu.items()], rows=rows, drows=drows,
                forks=[list(k) + [v] for k, v in fork_c.items()])


if __name__ == '__main__':
    files = sorted(glob.glob('/workspace/cl/ps_p13_*/graphs/*.json'))
    if len(sys.argv) > 1: files = files[:int(sys.argv[1])]
    with Pool(40, maxtasksperchild=4) as p: R = p.map(job, files, chunksize=1)
    json.dump(R, open('/workspace/cl/ideas/r2_caps_out.json', 'w'))
    print('done', len(R))
