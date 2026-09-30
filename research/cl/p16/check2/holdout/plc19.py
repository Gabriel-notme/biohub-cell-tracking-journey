"""check2/holdout (b): placebo / negative control for the two NEW P19-R families on the P15 graphs (family applied alone):
  dup  : start_trim(2.5) -> term_trim(3.5, no join) -> par_dup(3.5). Placebo: the same number of heads / tails / segment ends of the
         same lengths, deleted from randomly chosen tracks that the rule did not touch (heads: plain track starts, not B5 fork
         daughters; tails: plain track ends, not fork daughters; par: a head or tail of a random linear segment), with a >= 3-node
         remainder so a placebo never deletes a whole track.
  bd   : yx_border_stubs(margin 2, minlen 6). Placebo: size-matched fork-free components with < 6 nodes that are NOT border stubs.
usage: plc19.py <K> <outdir>"""
import os, sys, json, glob, time, traceback, zlib
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path[:0] = ['/workspace/cl', '/workspace/official/src', '/workspace/code', '/workspace/cl/ideas', '/workspace/cl/p16/deploy', '/workspace/p12ds']
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import warnings; warnings.filterwarnings('ignore')
B5 = {'hold36': '/workspace/runs/b5f_hold36', 'prev4': '/workspace/runs/b5f_prev4', 'audit32': '/workspace/sync3/runs/b5f_audit32',
      't127a': '/workspace/sync4/runs/b5f_t127a', 't127b': '/workspace/sync3/runs/b5f_t127b'}
SETS = list(B5)


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


def small_comps(nodes, edges, minlen=6):
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
        out.append(sorted(comp))
    return out


def pick_by_size(pool, sizes, rng):
    pool = list(pool); chosen = []; short = 0
    for sz in (rng.permutation(sizes).tolist() if len(sizes) else []):
        if not pool: short += sz; continue
        d = np.array([abs(len(c) - sz) for c in pool]); cand = np.flatnonzero(d == d.min())
        j = int(cand[rng.integers(len(cand))]); chosen.append(pool.pop(j))
    return chosen, short


def heads(nodes, edges, D, excl):
    """plain track starts: no parent, exactly one child, not a B5 fork daughter; returns start -> forward linear chain"""
    succ, par = topo(edges); res = {}
    for n in nodes:
        if n in par or len(succ.get(n, [])) != 1 or n in D or n in excl: continue
        ch = [n]; c = n
        while len(succ.get(c, [])) == 1: c = succ[c][0]; ch.append(c)
        res[n] = ch
    return res


def tails(nodes, edges, excl):
    """plain track ends: no child, has a parent that is not a fork; returns end -> backward linear chain"""
    succ, par = topo(edges); res = {}
    for n in nodes:
        if succ.get(n) or n not in par or len(succ.get(par[n], [])) == 2 or n in excl: continue
        ch = [n]; c = n
        while c in par and len(succ.get(par[c], [])) == 1: c = par[c]; ch.append(c)
        res[n] = ch
    return res


def job(args):
    s, f, K, outdir = args
    name = os.path.basename(f)[:-5]; op = '%s/%s.json' % (outdir, name)
    if os.path.exists(op): return name, 'cached'
    try:
        try:
            import numcodecs.blosc; numcodecs.blosc.use_threads = False
        except Exception:
            pass
        import evalx, p19_dup, p14_post, p19_edge_deploy
        t0 = time.time()
        refp = B5[s] + '/working/reference_graphs/%s.json' % name
        D = p19_dup.ref_fork_daughters(refp)
        n0, e0 = evalx.load_graph_json(f)
        # rule, family alone, with per-sub-rule items
        a1, b1, _ = p19_dup.start_trim(n0, e0, D, r=2.5, minlen=3, iters=5); rm_st = set(n0) - set(a1)
        a2, b2, _ = p14_post.term_trim(a1, b1, r=3.5, join=None); rm_tt = set(a1) - set(a2)
        a3, b3, _, _ = p19_dup.par_dup(a2, b2, r=3.5, minrun=3, which='shorter'); rm_par = set(a2) - set(a3)
        rm_dup = rm_st | rm_tt | rm_par
        c1, d1, _ = p19_edge_deploy.yx_border_stubs(n0, e0, shape_yx=(256, 256), minlen=6, margin=2.0); rm_bd = set(n0) - set(c1)
        L_st = [len(c) for c in comps_of(rm_st, e0)]; L_tt = [len(c) for c in comps_of(rm_tt, b1)]
        # par items: which end of their segment? (we just use head/tail 50/50 in the placebo)
        L_par = [len(c) for c in comps_of(rm_par, b2)]
        L_bd = [len(c) for c in comps_of(rm_bd, e0)]
        info = {'movie': name, 'set': s, 'L_st': L_st, 'L_tt': L_tt, 'L_par': L_par, 'L_bd': L_bd, 'n_dup': len(rm_dup), 'n_bd': len(rm_bd)}
        succ0, par0 = topo(e0)
        H = heads(n0, e0, D, rm_dup); T = tails(n0, e0, rm_dup)
        bd_pool = [c for c in small_comps(n0, e0) if not (set(c) & rm_bd)]
        V = [('p15', set()), ('dup', rm_dup), ('bd', rm_bd)]
        seed0 = zlib.crc32(name.encode()); pinfo = []
        for k in range(K):
            rng = np.random.default_rng([seed0, k, 7])
            used = set(rm_dup); drop = set(); short = 0
            hk = sorted(H); tk = sorted(T)
            for L in rng.permutation(L_st).tolist() if L_st else []:
                ok = [h for h in hk if len(H[h]) >= L + 3 and not (set(H[h][:L]) & used)]
                if not ok: short += L; continue
                h = ok[rng.integers(len(ok))]; drop.update(H[h][:L]); used.update(H[h][:L + 1])
            for L in rng.permutation(L_tt).tolist() if L_tt else []:
                ok = [t for t in tk if len(T[t]) >= L + 3 and not (set(T[t][:L]) & used)]
                if not ok: short += L; continue
                t = ok[rng.integers(len(ok))]; drop.update(T[t][:L]); used.update(T[t][:L + 1])
            for L in rng.permutation(L_par).tolist() if L_par else []:
                if rng.random() < 0.5:
                    ok = [h for h in hk if len(H[h]) >= L + 3 and not (set(H[h][:L]) & used)]
                    if not ok: short += L; continue
                    h = ok[rng.integers(len(ok))]; drop.update(H[h][:L]); used.update(H[h][:L + 1])
                else:
                    ok = [t for t in tk if len(T[t]) >= L + 3 and not (set(T[t][:L]) & used)]
                    if not ok: short += L; continue
                    t = ok[rng.integers(len(ok))]; drop.update(T[t][:L]); used.update(T[t][:L + 1])
            rng2 = np.random.default_rng([seed0, k, 8])
            ch, sh2 = pick_by_size(bd_pool, L_bd, rng2); pbd = {x for c in ch for x in c}
            V += [('Pdup%d' % k, drop), ('Pbd%d' % k, pbd)]
            pinfo.append({'k': k, 'dup_nodes': len(drop), 'dup_short': short, 'bd_nodes': len(pbd), 'bd_short': sh2})
        info['placebo'] = pinfo; info['pool'] = {'heads': len(H), 'tails': len(T), 'bd': len(bd_pool)}
        rows = []
        for tag, drop in V:
            nn, ee = sub(n0, e0, drop)
            r = evalx.score_movie(name, nn, ee); r['var'] = tag; r['set'] = s; rows.append(r)
        info['sec'] = time.time() - t0
        json.dump({'info': info, 'rows': rows}, open(op + '.tmp', 'w')); os.replace(op + '.tmp', op)
        return name, 'ok %.0fs' % info['sec']
    except Exception:
        return name, 'ERR ' + traceback.format_exc()[-2000:]


if __name__ == '__main__':
    K = int(sys.argv[1]); outdir = sys.argv[2]; flt = os.environ.get('FLT', '')
    os.makedirs(outdir, exist_ok=True)
    jobs = [(s, f, K, outdir) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s)) if flt in os.path.basename(f)]
    print(len(jobs), 'jobs', flush=True)
    if 'SHARD' in os.environ:  # sequential shard k/n in this process (no fork: avoids futex deadlocks seen with Pool)
        k, n = map(int, os.environ['SHARD'].split('/'))
        for j in jobs[k::n]: print(*job(j), flush=True)
    else:
        with Pool(int(os.environ.get('RULE_POOL', '6')), maxtasksperchild=4) as p:
            for name, st in p.imap_unordered(job, jobs, chunksize=1): print(name, st, flush=True)
