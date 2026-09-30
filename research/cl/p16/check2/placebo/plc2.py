"""chk2 placebo stage 2 (key: placebo): node- and structure-matched random deletions on P15, K draws per movie.
For every movie the deployed P19-R families are re-applied (cd, ff, st, tt, par, bd; dup = st+tt+par) and each removed piece is
typed (bridge / whole / head / tail, size). Each draw replaces every rule piece by a random piece of the SAME family-eligible
structure and size that the rules did NOT select (disjoint from all P19-R-removed nodes, not adjacent to other placebo pieces):
  cd  : B5 gap-bridge synthetic chains (exact cutdup definition) not selected by cutdup
  ff  : fork-free weakly connected components of the same size (< 6 nodes) not removed by any rule
  st  : track heads: root r not a B5-ref fork daughter, first k nodes single-child, forward linear length >= 3 after them
        (start_trim eligibility minus the 2.5 um duplicate test)
  tt  : track tails: last k nodes of a track, none a fork daughter, backward depth >= 3 (term_trim eligibility minus the 3.5 um test)
  par : head / tail / whole pieces of linear tracks of the same size (generic eligibility)
  bd  : fork-free components of the same size (< 6 nodes) anywhere in the FOV (border test removed)
Exact official scoring via lib.Scorer (reduced to GT-relevant components; num_pred_nodes of the full graph), verified per movie
against evalx.score_movie on the first NV draws.   usage: plc2.py K [movie-filter]"""
import os, sys, json, time, glob, zlib, traceback, itertools
sys.path.insert(0, '/workspace/cl/p16/check2/placebo')
from lib import *
from multiprocessing import Pool

OUT = '/workspace/cl/p16/check2/placebo/rows'
KEYS = ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes']
NV = 3
VARS = ['A', 'cd', 'ff', 'st', 'tt', 'par', 'bd', 'dup']


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
    trees = {}; selnode = set()
    for n in syn:
        t = int(nodes[n]['t'])
        if t not in trees: trees[t] = (ids_by_t[t], cKDTree(np.stack([pos[m] for m in ids_by_t[t]])))
        ns, tr = trees[t]
        if any(ns[k] != n for k in tr.query_ball_point(pos[n], rad)): selnode.add(n)
    fe = [e for e in edges if fl[(int(e['source_id']), int(e['target_id']))] in ('gc', 'g2')]
    return [(c, any(x in selnode for x in c)) for c in comps_of(syns, fe)]


def build_pools(nodes, edges, D, excl, kmax=6):
    succ, par = topo(edges); lab, comps = wcc(nodes, edges)
    pools = {}
    pools[('cd', 'bridge')] = [c for c, sel in syn_chains(nodes, edges) if not sel and not any(x in excl for x in c)]
    whole = [sorted(c) for c in comps if len(c) <= kmax and not any(len(succ.get(u, [])) >= 2 for u in c) and not any(x in excl for x in c)]
    pools[('small', 'whole')] = [c for c in whole if len(c) < 6]
    pools[('par', 'whole')] = whole
    hst = []; hgen = []; tst = []; tgen = []
    for r in nodes:
        if r in par: continue
        ch = []; c = r; okst = True
        for k in range(1, kmax + 1):
            if len(succ.get(c, [])) != 1 or c in excl: break
            nx = succ[c][0]
            ch = ch + [c]
            hgen.append(ch)
            if okst and (c in D or len(succ.get(nx, [])) != 1): okst = False
            if okst: hst.append(ch)
            c = nx
    for l in nodes:
        if succ.get(l) or l not in par: continue
        ch = []; c = l; oktt = True
        for k in range(1, kmax + 1):
            if c not in par or c in excl: break
            p = par[c]
            if len(succ.get(p, [])) != 1: break
            ch = ch + [c]
            tgen.append(ch)
            if oktt and p not in par: oktt = False
            if oktt: tst.append(ch)
            c = p
    pools[('st', 'head')] = hst; pools[('par', 'head')] = hgen; pools[('tt', 'tail')] = tst; pools[('par', 'tail')] = tgen
    bysz = {}
    for key, lst in pools.items():
        d = defaultdict(list)
        for c in lst: d[len(c)].append(c)
        bysz[key] = d
    return bysz


def pool_key(fam, typ):
    if fam == 'cd': return ('cd', 'bridge')
    if fam in ('ff', 'bd'): return ('small', 'whole')
    if fam == 'st': return ('st', 'head')
    if fam == 'tt': return ('tt', 'tail')
    if fam == 'par': return ('par', typ if typ in ('head', 'tail', 'whole') else 'head')
    raise ValueError(fam)


def draw(targets, bysz, adj, rng):
    """targets: [(fam, typ, size)] in rule order. Returns {fam: set}, {fam: shortfall nodes (target - placebo)}"""
    U = set(); out = {f: set() for f in FAMS}; short = {f: 0 for f in FAMS}
    for fam in FAMS:
        tl = [t for t in targets if t[0] == fam]
        for i in rng.permutation(len(tl)):
            _, typ, sz = tl[i]
            d = bysz[pool_key(fam, typ)]
            sizes = sorted(d, key=lambda s: (abs(s - sz), -s))
            got = None
            for s2 in sizes:
                lst = d[s2]
                if not lst: continue
                if len(lst) <= 64: order = rng.permutation(len(lst))
                else: order = itertools.chain(rng.integers(0, len(lst), 64), (int(q) for q in rng.permutation(len(lst))))
                for j in order:
                    c = lst[j]
                    if any(x in U for x in c) or any(y in U for x in c for y in adj.get(x, ())): continue
                    got = c; break
                if got is not None: break
            if got is None: short[fam] += sz; continue
            short[fam] += sz - len(got)
            U.update(got); out[fam].update(got)
    return out, short


def job(args):
    s, f, K = args
    name = os.path.basename(f)[:-5]; op = '%s/%s.json' % (OUT, name)
    if os.path.exists(op): return name, 'cached'
    try:
        import evalx
        from tracking_cellmot.metrics import per_sample_metrics, EvaluationResult
        t0 = time.time()
        n15, e15 = load(f); refp = B5[s] + '/working/reference_graphs/%s.json' % name
        steps, (nf, ef), D = apply_rules(n15, e15, refp)
        rm = {fam: r for fam, _, _, r in steps}
        excl = set(n15) - set(nf)
        targets = [(fam, p['typ'], p['size']) for fam, a, b, r in steps for p in piece_types(a, b, r)]
        sc = Scorer(name, n15, e15)
        adj = defaultdict(list)
        for e in e15:
            a, b = int(e['source_id']), int(e['target_id']); adj[a].append(b); adj[b].append(a)
        bysz = build_pools(n15, e15, D, excl)
        cache = {}
        nrl = [0]

        def counts(drop):
            key = frozenset(x for x in drop if x in sc.R0)
            if key not in cache:
                r = sc.score(key); nrl[0] += 1
                cache[key] = tuple(r[k] for k in KEYS[:6]) + (r['node_recall'],)
            c = cache[key]
            return list(c[:6]) + [len(n15) - len(drop)]

        obs ={'p15': counts(set()), 'P19': counts(excl)}
        for fam in FAMS: obs[fam] = counts(rm[fam])
        obs['dup'] = counts(rm['st'] | rm['tt'] | rm['par'])
        tg = {fam: sum(t[2] for t in targets if t[0] == fam) for fam in FAMS}
        seed0 = zlib.crc32(('chk2placebo:' + name).encode())
        draws = {v: [] for v in VARS}; shorts = []; ver = []
        for k in range(K):
            rng = np.random.default_rng([seed0, k])
            pf, sh = draw(targets, bysz, adj, rng)
            allp = set().union(*pf.values())
            vs = {'A': allp, 'dup': pf['st'] | pf['tt'] | pf['par']}
            for fam in FAMS: vs[fam] = pf[fam]
            for v in VARS: draws[v].append(counts(vs[v]))
            shorts.append([sh[fam] for fam in FAMS])
            if k < NV and (allp & sc.R0):  # verification of the reduced scorer against the official full path
                fn_, fe_ = sub(n15, e15, allp); r = evalx.score_movie(name, fn_, fe_)
                ver.append([r[kk] for kk in KEYS] == draws['A'][-1])
        # verify obs P19 against the full official path too
        r = evalx.score_movie(name, nf, ef); ver.append([r[kk] for kk in KEYS] == obs['P19'])
        info = {'movie': name, 'set': s, 'n15': len(n15), 'n_total': sc.nt, 'K': K, 'tg': tg, 'targets': targets,
                'pool': {'%s/%s' % k: {int(a): len(b) for a, b in v.items()} for k, v in bysz.items()},
                'ver': ver, 'nscore': nrl[0], 'R0': len(sc.R0), 'sec': time.time() - t0}
        json.dump({'info': info, 'obs': obs, 'draws': draws, 'short': shorts}, open(op + '.tmp', 'w')); os.replace(op + '.tmp', op)
        return name, 'ok ver=%s nscore=%d %.0fs' % (all(ver), nrl[0], time.time() - t0)
    except Exception:
        return name, 'ERR ' + traceback.format_exc()[-1500:]


if __name__ == '__main__':
    K = int(sys.argv[1]); flt = sys.argv[2] if len(sys.argv) > 2 else ''
    if len(sys.argv) > 3: OUT = sys.argv[3]
    os.makedirs(OUT, exist_ok=True)
    jobs = [(s, f, K) for s in SETS for f in sorted(glob.glob(P15 % s + '/*.json')) if flt in os.path.basename(f)]
    # biggest movies first for load balance
    jobs.sort(key=lambda j: -os.path.getsize(j[1]))
    print(len(jobs), 'jobs', flush=True)
    with Pool(int(os.environ.get('RULE_POOL', '6')), maxtasksperchild=8) as p:
        for name, st in p.imap_unordered(job, jobs, chunksize=1): print(name, st, flush=True)
