"""Placebo / negative-control check for P17 (key: placebo). Writes only under /workspace/cl/p16/check/placebo/.
usage: plc.py <K> [movie-filter] [outdir]"""
import os, sys, json, zlib, time, glob, traceback
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/cl/p16/deploy')
from collections import defaultdict
import numpy as np
from multiprocessing import Pool

SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
B5 = {'hold36': '/workspace/runs/b5f_hold36', 'prev4': '/workspace/runs/b5f_prev4', 'audit32': '/workspace/sync3/runs/b5f_audit32',
      't127a': '/workspace/sync4/runs/b5f_t127a', 't127b': '/workspace/sync3/runs/b5f_t127b'}
P15 = '/workspace/cl/ps_p15_%s/graphs'; P17 = '/workspace/cl/p16/ps_p17_%s/graphs'
S = np.array([1.625, .40625, .40625])
OUT = '/workspace/cl/p16/check/placebo/rows'


def load(p):
    d = json.load(open(p)); return {int(k): v for k, v in d['nodes'].items()}, d['edges']


def sub(nodes, edges, drop):
    return ({k: v for k, v in nodes.items() if k not in drop},
            [e for e in edges if int(e['source_id']) not in drop and int(e['target_id']) not in drop])


def topo(edges):
    succ = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a
    return succ, par


def comps_of(nset, edges):
    adj = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id'])
        if a in nset and b in nset: adj[a].append(b); adj[b].append(a)
    seen = set(); out = []
    for n in sorted(nset):
        if n in seen: continue
        c = []; st = [n]; seen.add(n)
        while st:
            u = st.pop(); c.append(u)
            for v in adj[u]:
                if v not in seen: seen.add(v); st.append(v)
        out.append(sorted(c))
    return out


def syn_chains(nodes, edges, rad=3.2):
    """B5 gap-bridge synthetic chains exactly as p17_post.cutdup defines them; returns [(chain, selected_by_rule)]"""
    from scipy.spatial import cKDTree
    succ, par = topo(edges); fl = {}
    for e in edges:
        fl[(int(e['source_id']), int(e['target_id']))] = 'gc' if 'gap_closed' in e else ('g2' if 'gap2_recovered' in e else 'o')
    syn = [n for n, v in nodes.items() if n in par and fl.get((par[n], n)) in ('gc', 'g2') and len(succ.get(n, [])) == 1
           and fl.get((n, succ[n][0])) in ('gc', 'g2') and ('gap_synthetic' in v or fl.get((par[n], n)) == 'g2')]
    syns = set(syn)
    if not syns: return []
    pos = {n: np.array([float(v['z']), float(v['y']), float(v['x'])]) * S for n, v in nodes.items()}
    ids_by_t = defaultdict(list)
    for n, v in nodes.items(): ids_by_t[int(v['t'])].append(n)
    trees = {}
    selnode = set()
    for n in syn:
        t = int(nodes[n]['t'])
        if t not in trees: trees[t] = (ids_by_t[t], cKDTree(np.stack([pos[m] for m in ids_by_t[t]])))
        ns, tr = trees[t]
        if any(ns[k] != n for k in tr.query_ball_point(pos[n], rad)): selnode.add(n)
    fe = [e for e in edges if fl[(int(e['source_id']), int(e['target_id']))] in ('gc', 'g2')]
    return [(c, any(x in selnode for x in c)) for c in comps_of(syns, fe)]


def small_comps(nodes, edges, D, minlen=6):
    succ, par = topo(edges); adj = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); adj[a].append(b); adj[b].append(a)
    seen = set(); out = []
    for n in sorted(nodes):
        if n in seen: continue
        comp = []; st = [n]; seen.add(n)
        while st:
            u = st.pop(); comp.append(u)
            for v in adj.get(u, []):
                if v not in seen: seen.add(v); st.append(v)
        if len(comp) >= minlen or any(len(succ.get(u, [])) >= 2 for u in comp): continue
        out.append((sorted(comp), any(u in D and u not in par for u in comp)))
    return out


def sb_selected_forks(edges, maxk=2):
    succ, par = topo(edges); sel = {}
    for p, ch in list(succ.items()):
        if len(ch) != 2: continue
        for c in ch:
            br = [c]; n = c; ok = True
            while True:
                nx = succ.get(n, [])
                if len(nx) == 0: break
                if len(nx) == 2 or len(br) > maxk: ok = False; break
                n = nx[0]; br.append(n)
            if ok and len(br) <= maxk + 1: sel[p] = br; break
    return sel


def tails(nodes, edges, L, excl):
    """dead-end tail segments of L nodes of a LINEAR track (anchor above the segment exists, is not a fork, and itself has a parent)"""
    succ, par = topo(edges); res = []
    for e in sorted(nodes):
        if succ.get(e): continue
        seg = [e]; cur = e; ok = True
        while len(seg) < L:
            p = par.get(cur)
            if p is None or len(succ.get(p, [])) != 1: ok = False; break
            seg.append(p); cur = p
        if not ok: continue
        a = par.get(cur)
        if a is None or len(succ.get(a, [])) != 1 or a not in par: continue
        if any(x in excl for x in seg) or a in excl: continue
        res.append(seg)
    return res


def fork_cuts(nodes, edges, L, excl):
    """forks NOT selected by shortbranch: cut the first L nodes of one daughter branch (branch continues -> remainder orphaned)"""
    succ, par = topo(edges); sel = sb_selected_forks(edges); res = defaultdict(list)
    for p in sorted(succ):
        ch = succ[p]
        if len(ch) != 2 or p in sel or p in excl or any(c in excl for c in ch): continue
        for c in ch:
            seg = [c]; cur = c; ok = len(succ.get(c, [])) == 1
            while ok and len(seg) < L:
                cur = succ[cur][0]; seg.append(cur)
                if len(succ.get(cur, [])) != 1: ok = False
            if ok and not any(x in excl for x in seg): res[p].append(seg)
    return res


def pick_by_size(pool, sizes, rng):
    """pool: list of node lists; sizes: target sizes. Same size if possible, else closest. returns chosen lists, shortfall nodes"""
    pool = list(pool); chosen = []; short = 0
    for sz in (rng.permutation(sizes).tolist() if len(sizes) else []):
        if not pool: short += sz; continue
        d = np.array([abs(len(c) - sz) for c in pool]); cand = np.flatnonzero(d == d.min())
        j = int(cand[rng.integers(len(cand))]); chosen.append(pool.pop(j))
    return chosen, short


def placebo(nodes, edges, D, excl, tgt, rng1, rng2, mode):
    info = {}
    # 1) gap-bridge chains not selected by cutdup
    pool = [c for c, sel in syn_chains(nodes, edges) if not sel and not any(x in excl for x in c)]
    ch, sh = pick_by_size(pool, tgt['cd'], rng1)
    pcd = {x for c in ch for x in c}; info.update(cd_pool=len(pool), cd_nodes=len(pcd), cd_short=sh)
    n1, e1 = sub(nodes, edges, pcd)
    # 2) small fork-free components not selected by forkfrag
    pool = [c for c, sel in small_comps(n1, e1, D) if not sel and not any(x in excl for x in c)]
    ch, sh = pick_by_size(pool, tgt['ff'], rng1)
    pff = {x for c in ch for x in c}; info.update(ff_pool=len(pool), ff_nodes=len(pff), ff_short=sh)
    n2, e2 = sub(n1, e1, pff)
    # 3) shortbranch analogue
    psb = set(); sh = 0; used = set(); forks_cut = []
    if mode == 'A':
        pools = {L: tails(n2, e2, L, excl) for L in sorted(set(tgt['sb']))}
        info['sb_pool'] = sum(len(v) for v in pools.values())
        for L in (rng2.permutation(tgt['sb']).tolist() if len(tgt['sb']) else []):
            cand = [s for s in pools[L] if not any(x in used for x in s)]
            if not cand: sh += L; continue
            s = cand[rng2.integers(len(cand))]; psb.update(s); used.update(s)
    else:
        pools = {L: fork_cuts(n2, e2, L, excl) for L in sorted(set(tgt['sb']))}
        info['sb_pool'] = sum(len(v) for v in pools.values())
        for L in (rng2.permutation(tgt['sb']).tolist() if len(tgt['sb']) else []):
            fk = [p for p in sorted(pools[L]) if p not in used]
            if not fk: sh += L; continue
            p = fk[rng2.integers(len(fk))]; opts = pools[L][p]; s = opts[rng2.integers(len(opts))]
            if any(x in used for x in s): sh += L; continue
            psb.update(s); used.update(s); used.add(p); forks_cut.append(p)
    info.update(sb_nodes=len(psb), sb_short=sh)
    return pcd, pff, psb, info, forks_cut


def fork_status(name, nodes, edges):
    import evalx
    from tracking_cellmot.division_metrics import score_divisions
    gt, nt = evalx.load_gt(name); pred, mp = evalx.to_graph(nodes, edges)
    r = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mp.items()}
    return {inv[x] for x in r.tp_forks}, {inv[x] for x in r.fp_forks}


def job(args):
    s, f, K = args
    name = os.path.basename(f)[:-5]; op = '%s/%s.json' % (OUT, name)
    if os.path.exists(op): return name, 'cached'
    try:
        import evalx, p17_post
        t0 = time.time()
        n15, e15 = load(f); n17, e17 = load('%s/%s.json' % (P17 % s, name))
        refp = B5[s] + '/working/reference_graphs/%s.json' % name
        d = json.load(open(refp)); ro = defaultdict(list)
        for e in d['edges']: ro[int(e['source_id'])].append(int(e['target_id']))
        D = {c for a, cs in ro.items() if len(cs) >= 2 for c in cs}
        # sequential rule application (deployed code)
        a1, b1, _ = p17_post.cutdup(n15, e15); rm_cd = set(n15) - set(a1)
        a2, b2, _ = p17_post.forkfrag(a1, b1, refp); rm_ff = set(a1) - set(a2)
        r3 = p17_post.short_branch(a2, b2); a3, b3 = r3[0], r3[1]; rm_sb = set(a2) - set(a3)
        info = {'movie': name, 'set': s, 'n15': len(n15), 'n17': len(n17)}
        info['p17_eq_rules'] = (set(a3) == set(n17)) and ({(int(e['source_id']), int(e['target_id'])) for e in b3} == {(int(e['source_id']), int(e['target_id'])) for e in e17})
        info['p17_subset_p15'] = set(n17) <= set(n15)
        excl = set(n15) - set(n17)
        tgt = {'cd': [len(c) for c in comps_of(rm_cd, e15)], 'ff': [len(c) for c in comps_of(rm_ff, b1)], 'sb': [len(c) for c in comps_of(rm_sb, b2)]}
        info['tgt'] = tgt; info['rm'] = {'cd': len(rm_cd), 'ff': len(rm_ff), 'sb': len(rm_sb), 'all': len(excl)}
        succ15, _ = topo(e15); forks15 = {p for p, c in succ15.items() if len(c) == 2}
        tpf, fpf = fork_status(name, n15, e15)
        info['forks15'] = len(forks15); info['tpf'] = sorted(tpf); info['fpf'] = sorted(fpf)

        def lost_forks(drop):
            nn, ee = sub(n15, e15, drop); sc, _ = topo(ee)
            return sorted(p for p in forks15 if p in nn and len(sc.get(p, [])) < 2) + sorted(p for p in forks15 if p not in nn)
        info['lost_forks'] = {'cd': lost_forks(rm_cd), 'ff': lost_forks(rm_ff), 'sb': lost_forks(rm_sb), 'all': lost_forks(excl)}
        V = [('p15', set()), ('p17file', None), ('r_cd', rm_cd), ('r_cdff', rm_cd | rm_ff), ('r_ffonly', rm_ff), ('r_sbonly', rm_sb)]
        pinfo = []
        seed0 = zlib.crc32(name.encode())
        for k in range(K):
            rng1 = np.random.default_rng([seed0, k, 1]); rng2a = np.random.default_rng([seed0, k, 2]); rng2b = np.random.default_rng([seed0, k, 3])
            pcd, pff, psbA, iA, _ = placebo(n15, e15, D, excl, tgt, rng1, rng2a, 'A')
            rng1 = np.random.default_rng([seed0, k, 1])
            pcd2, pff2, psbB, iB, fc = placebo(n15, e15, D, excl, tgt, rng1, rng2b, 'B')
            assert pcd2 == pcd and pff2 == pff
            V += [('A%d' % k, pcd | pff | psbA), ('B%d' % k, pcd | pff | psbB), ('Acd%d' % k, pcd), ('Aff%d' % k, pff), ('Asb%d' % k, psbA), ('Bsb%d' % k, psbB)]
            pinfo.append({'k': k, 'A': iA, 'B': iB, 'B_forks_cut': fc,
                          'A_lost_forks': lost_forks(pcd | pff | psbA), 'B_lost_forks': lost_forks(pcd | pff | psbB)})
        info['placebo'] = pinfo
        rows = []
        for tag, drop in V:
            if drop is None: nn, ee = n17, e17
            else: nn, ee = sub(n15, e15, drop)
            r = evalx.score_movie(name, nn, ee); r['var'] = tag; r['set'] = s; rows.append(r)
        info['sec'] = time.time() - t0
        json.dump({'info': info, 'rows': rows}, open(op + '.tmp', 'w')); os.replace(op + '.tmp', op)
        return name, 'ok %.0fs' % info['sec']
    except Exception:
        return name, 'ERR ' + traceback.format_exc()[-1500:]


if __name__ == '__main__':
    K = int(sys.argv[1]); flt = sys.argv[2] if len(sys.argv) > 2 else ''
    if len(sys.argv) > 3: OUT = sys.argv[3]
    os.makedirs(OUT, exist_ok=True)
    jobs = [(s, f, K) for s in SETS for f in sorted(glob.glob(P15 % s + '/*.json')) if flt in os.path.basename(f)]
    print(len(jobs), 'jobs', flush=True)
    with Pool(int(os.environ.get('RULE_POOL', '6')), maxtasksperchild=4) as p:
        for name, st in p.imap_unordered(job, jobs, chunksize=1): print(name, st, flush=True)

