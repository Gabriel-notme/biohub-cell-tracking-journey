"""Division lens (read-only): per P13 fork and per GT division, official categories + origin + geometry.
Writes /workspace/cl/ideas/div_lens_rows.json"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}


def chain_len(x, ch, maxl=200):
    L = 1
    while len(ch.get(x, [])) == 1 and L < maxl: x = ch[x][0]; L += 1
    return L, len(ch.get(x, []))  # length, out-degree at end (0 = track end, 2 = next fork)


def hist_len(x, ch, par, maxl=200):
    h = 0
    while x in par and len(ch[par[x]]) == 1 and h < maxl: x = par[x]; h += 1
    return h, (x in par)  # history, whether it stops at a fork (True) or a root (False)


def job(a):
    s, f = a
    name = Path(f).stem
    import evalx, zarr
    from tracking_cellmot.division_metrics import score_divisions, _pred_division_fork_sets, _match_full, _matched_node_attrs
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    bn, be = evalx.load_graph_json(B5[s] + '/' + name + '.json')
    bch = defaultdict(list)
    for e in be: bch[int(e['source_id'])].append(int(e['target_id']))
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    ev, cross, mal = _pred_division_fork_sets(pred, gt, evalx.SCALE, 7.)
    mp = _match_full(pred, gt, evalx.SCALE, 7.)
    ma = _matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    g2p = {g: p for p, g in p2g.items()}
    tp = {inv[int(x)] for x in res.tp_forks}; fp = {inv[int(x)] for x in res.fp_forks}
    ev = {inv[int(x)] for x in ev}; cross = {inv[int(x)] for x in cross}; mal = {inv[int(x)] for x in mal}
    # GT structure
    z = zarr.open_group('/workspace/data/train/%s.geff' % name, mode='r')
    gids = np.asarray(z['nodes/ids']).tolist(); GP = np.stack([np.asarray(z['nodes/props/%s/values' % k]) for k in 'zyx'], 1) * S
    GT_ = np.asarray(z['nodes/props/t/values']).tolist(); GE = np.asarray(z['edges/ids']).tolist()
    gpos = dict(zip(gids, GP)); gt_t = dict(zip(gids, GT_)); gch = defaultdict(list); gpar = {}
    for u, v in GE: gch[u].append(v); gpar[v] = u
    gcomp = {}
    for sd in gids:
        if sd in gcomp: continue
        gcomp[sd] = sd; st = [sd]
        while st:
            c = st.pop()
            for nb in gch.get(c, []) + ([gpar[c]] if c in gpar else []):
                if nb not in gcomp: gcomp[nb] = sd; st.append(nb)
    gdiv = [d for d in gch if len(gch[d]) >= 2]
    ch = defaultdict(list); par = {}; eattr = {}
    for e in edges:
        u, v = int(e['source_id']), int(e['target_id']); ch[u].append(v); par[v] = u; eattr[(u, v)] = e
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}; t = {n: int(v['t']) for n, v in nodes.items()}
    forks = [n for n in ch if len(ch[n]) >= 2]
    rows = []
    for d in forks:
        kids = ch[d][:2]
        dc = any('div_complete' in eattr[(d, k)] for k in kids)
        origin = 'dc' if dc else ('b5' if sorted(bch.get(d, [])) == sorted(kids) else 'mod')
        flags = sorted({k2 for k in kids for k2 in eattr[(d, k)] if k2 not in ('source_id', 'target_id', 'edge_prob', 'distance_um', 'motion_distance_um', 'motion_pass', 'motion_relinked')})
        bl = [chain_len(k, ch) for k in kids]; h = hist_len(d, ch, par)
        # nearest GT division in space-time (divider position vs fork parent)
        best = None
        for g in gdiv:
            dt = t[d] - gt_t[g]
            if abs(dt) > 5: continue
            dist = float(np.linalg.norm(gpos[g] - pos[d]))
            if best is None or dist + 3 * abs(dt) < best[2] + 3 * abs(best[0]): best = (dt, g, dist)
        lab = 'TP' if d in tp else ('FP' if d in fp else 'U')
        cat = []
        if d in ev: cat.append('ev')
        if d in cross: cat.append('cross')
        if d in mal: cat.append('mal')
        if lab == 'FP' and not cat: cat.append('considered')
        gp = p2g.get(d); gk = [p2g.get(k) for k in kids]
        rows.append(dict(kind='fork', set=s, movie=name, p=d, a=kids[0], b=kids[1], t=t[d], lab=lab, cat=cat, origin=origin, flags=flags,
                         la=bl[0][0], la_end=bl[0][1], lb=bl[1][0], lb_end=bl[1][1], hist=h[0], hist_fork=h[1],
                         dab=float(np.linalg.norm(pos[kids[0]] - pos[kids[1]])), dpa=float(np.linalg.norm(pos[kids[0]] - pos[d])), dpb=float(np.linalg.norm(pos[kids[1]] - pos[d])),
                         p_match=gp is not None, p_gt_out=len(gch.get(gp, [])) if gp is not None else -1, p_gt_in=int(gp in gpar) if gp is not None else -1,
                         kid_match=[k is not None for k in gk], kid_comp_same=(gk[0] is not None and gk[1] is not None and gcomp[gk[0]] == gcomp[gk[1]]),
                         kid_same_as_pcomp=[(k is not None and gp is not None and gcomp[k] == gcomp[gp]) for k in gk],
                         near_gdiv=None if best is None else dict(dt=best[0], dist=best[2], gdiv_id=best[1], tp=res.scores.get(best[1]))))
    # GT divisions
    for g in gdiv:
        kids = gch[g][:2]
        mpd = g2p.get(g); mgp = g2p.get(gpar.get(g)) if g in gpar else None; mk = [g2p.get(k) for k in kids]
        # pred forks nearby (dt in [-3,3], parent within 12 um of GT divider)
        near = []
        for d in forks:
            dt = t[d] - gt_t[g]
            if abs(dt) <= 3:
                dist = float(np.linalg.norm(gpos[g] - pos[d]))
                if dist <= 12: near.append(dict(p=d, dt=dt, dist=round(dist, 2), lab='TP' if d in tp else ('FP' if d in fp else 'U'),
                                                origin='dc' if any('div_complete' in eattr[(d, k)] for k in ch[d][:2]) else ('b5' if sorted(bch.get(d, [])) == sorted(ch[d][:2]) else 'mod')))
        # what the pred does at the divider: out-degree of matched divider node, and whether kid matches are in its successors
        rows.append(dict(kind='gdiv', set=s, movie=name, g=g, t=gt_t[g], tp=res.scores.get(g), gpar=int(g in gpar), g_hist=hist_len(g, gch, gpar)[0],
                         gk_len=[chain_len(k, gch)[0] for k in kids], d_kids_um=float(np.linalg.norm(gpos[kids[0]] - gpos[kids[1]])),
                         m_div=mpd is not None, m_div_out=len(ch.get(mpd, [])) if mpd is not None else -1, m_gpar=mgp is not None,
                         m_kids=[k is not None for k in mk],
                         kids_under_div=[(k is not None and mpd is not None and par.get(k) == mpd) for k in mk],
                         kid_pars_out=[(len(ch.get(par[k], [])) if (k is not None and k in par) else -1) for k in mk],
                         kid_has_par=[(k is not None and k in par) for k in mk],
                         near=near))
    return rows


if __name__ == '__main__':
    jobs = [(s, f) for s in B5 for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(12) as p: R = [r for rs in p.map(job, jobs, chunksize=1) for r in rs]
    json.dump(R, open('/workspace/cl/ideas/div_lens_rows.json', 'w'), default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print('rows', len(R))
