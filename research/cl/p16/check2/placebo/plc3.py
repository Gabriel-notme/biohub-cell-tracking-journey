"""chk2 placebo stage 3 (key: placebo): IN-CONTEXT per-family placebo. The per-family 'alone on P15' test (plc2) is distorted because
st/tt pieces are chosen after cutdup has cut tracks (a tail in the P19-R pipeline can be a track interior in P15). Here each family is
tested in the graph it actually operates on:
  seq_<f> : base = graph just before family f in the deployed order (cd, ff, st, tt, par, bd); obs = base minus rule pieces of f;
            placebo = base minus random matched pieces (same family eligibility, drawn on the base graph, disjoint from ALL P19-R nodes)
  seq_dup : base = graph after cd+ff; obs/placebo = st+tt+par pieces
  loo_<f> : base = P19-R plus back the pieces of f (leave-one-family-out); obs = P19-R; placebo = base minus random matched pieces
usage: plc3.py K [movie-filter]"""
import os, sys, json, time, glob, zlib, traceback
sys.path.insert(0, '/workspace/cl/p16/check2/placebo')
from lib import *
from plc2 import build_pools, draw, KEYS
from multiprocessing import Pool

OUT = os.environ.get('PLC3_OUT', '/workspace/cl/p16/check2/placebo/rows3')
CGROUPS = [['seq_cd', 'seq_ff', 'seq_st'], ['seq_tt', 'seq_par', 'seq_bd', 'seq_dup'], ['loo_cd', 'loo_ff'], ['loo_st', 'loo_tt', 'loo_dup']]


def job(args):
    s, f, K, gi = args
    name = os.path.basename(f)[:-5]; op = '%s/%s__g%d.json' % (OUT, name, gi)
    if os.path.exists(op): return name, gi, 'cached'
    try:
        t0 = time.time()
        n15, e15 = load(f); refp = B5[s] + '/working/reference_graphs/%s.json' % name
        steps, (nf, ef), D = apply_rules(n15, e15, refp)
        rm = {fam: r for fam, _, _, r in steps}; pre = {fam: (a, b) for fam, a, b, _ in steps}
        allrm = set(n15) - set(nf)
        tg = {fam: [(fam, p['typ'], p['size']) for p in piece_types(a, b, r)] for fam, a, b, r in steps}
        res = {}
        seed0 = zlib.crc32(('chk2placebo3:' + name).encode())
        for ci, ctx in enumerate(CGROUPS[gi]):
            kind, fam = ctx.split('_')
            fams = ['st', 'tt', 'par'] if fam == 'dup' else [fam]
            rmf = set().union(*[rm[x] for x in fams])
            if kind == 'seq':
                bn, be = pre['st'] if fam == 'dup' else pre[fam]
            else:
                bn, be = sub(n15, e15, allrm - rmf)
            targets = [t for x in fams for t in tg[x]]
            sc = Scorer(name, bn, be)
            bysz = build_pools(bn, be, D, allrm)
            adj = defaultdict(list)
            for e in be:
                a, b = int(e['source_id']), int(e['target_id']); adj[a].append(b); adj[b].append(a)
            cache = {}

            def counts(drop):
                key = frozenset(x for x in drop if x in sc.R0)
                if key not in cache:
                    r = sc.score(key); cache[key] = tuple(r[k] for k in KEYS[:6])
                return list(cache[key]) + [sc.N - len(drop)]
            assert rmf <= set(bn)
            base = counts(set()); obs = counts(rmf)
            dr = []; sh = []
            for k in range(K):
                rng = np.random.default_rng([seed0, zlib.crc32(ctx.encode()), k])
                pf, s_ = draw(targets, bysz, adj, rng)
                pl = set().union(*pf.values())
                dr.append(counts(pl)); sh.append(sum(s_.values()))
            res[ctx] = {'base': base, 'obs': obs, 'draws': dr, 'short': sh, 'n_targets': len(targets), 'nscore': len(cache)}
        # P19-R file identity check for the loo contexts: base minus rmf must equal P19-R exactly
        json.dump({'movie': name, 'set': s, 'n_total': sc.nt, 'K': K, 'res': res, 'sec': time.time() - t0}, open(op + '.tmp', 'w'))
        os.replace(op + '.tmp', op)
        return name, gi, 'ok %.0fs' % (time.time() - t0)
    except Exception:
        return name, gi, 'ERR ' + traceback.format_exc()[-1500:]


if __name__ == '__main__':
    K = int(sys.argv[1]); flt = sys.argv[2] if len(sys.argv) > 2 else ''
    os.makedirs(OUT, exist_ok=True)
    R0 = {}
    for fch in glob.glob('/workspace/cl/p16/check2/placebo/char/*.json'):
        d = json.load(open(fch)); R0[d['info']['movie']] = d['info']['R0']
    jobs = [(s, f, K, gi) for s in SETS for f in sorted(glob.glob(P15 % s + '/*.json')) if flt in os.path.basename(f) for gi in range(len(CGROUPS))]
    jobs.sort(key=lambda j: -R0.get(os.path.basename(j[1])[:-5], 0))
    print(len(jobs), 'jobs', flush=True)
    with Pool(int(os.environ.get('RULE_POOL', '6')), maxtasksperchild=8) as p:
        for name, gi, st in p.imap_unordered(job, jobs, chunksize=1): print(name, gi, st, flush=True)
