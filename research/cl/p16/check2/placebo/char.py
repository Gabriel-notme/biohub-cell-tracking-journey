"""chk2 placebo stage 1: reproduce P19-R per family on P15, type the removed pieces, verify the reduced scorer against the official
full-graph scorer (fresh GT load) on P15 and P19-R.  usage: char.py [movie-filter]"""
import os, sys, json, time, glob, traceback
sys.path.insert(0, '/workspace/cl/p16/check2/placebo')
from lib import *
from multiprocessing import Pool

OUT = '/workspace/cl/p16/check2/placebo/char'
KEYS = ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes']


def job(args):
    s, f = args
    name = os.path.basename(f)[:-5]; op = '%s/%s.json' % (OUT, name)
    if os.path.exists(op): return name, 'cached'
    try:
        import evalx, p19r
        t0 = time.time()
        n15, e15 = load(f); refp = B5[s] + '/working/reference_graphs/%s.json' % name
        steps, (nf, ef), D = apply_rules(n15, e15, refp)
        dn, de, _ = p19r.apply(n15, e15, name=name, set=s)
        info = {'movie': name, 'set': s, 'n15': len(n15)}
        info['eq_deployed'] = (set(dn) == set(nf)) and ekey(de) == ekey(ef)
        info['edges_subset'] = ekey(ef) <= ekey(e15)
        info['rm'] = {fam: len(rm) for fam, _, _, rm in steps}
        sc = Scorer(name, n15, e15)
        info['R0'] = len(sc.R0); info['matchable'] = len(sc.matchable)
        pcs = {}
        for fam, a, b, rm in steps:
            P = piece_types(a, b, rm)
            for p in P:
                p['inR0'] = any(x in sc.R0 for x in p['nodes']); p['nmatch'] = sum(x in sc.matchable for x in p['nodes'])
            pcs[fam] = P
        info['pieces'] = pcs
        allrm = set(n15) - set(nf)
        t1 = time.time()
        rows = {'p15_red': sc.score(set()), 'p19_red': sc.score(allrm)}
        t2 = time.time()
        for fam, _, _, rm in steps: rows['fam_' + fam] = sc.score(rm)
        t3 = time.time()
        rows['p15_full'] = evalx.score_movie(name, n15, e15); rows['p19_full'] = evalx.score_movie(name, nf, ef)
        t4 = time.time()
        info['exact'] = all(rows['p15_red'][k] == rows['p15_full'][k] for k in KEYS) and all(rows['p19_red'][k] == rows['p19_full'][k] for k in KEYS)
        info['t'] = {'rules': t1 - t0, 'red2': t2 - t1, 'fam6': t3 - t2, 'full2': t4 - t3}
        json.dump({'info': info, 'rows': rows}, open(op + '.tmp', 'w')); os.replace(op + '.tmp', op)
        return name, 'ok exact=%s eq=%s %.1fs' % (info['exact'], info['eq_deployed'], time.time() - t0)
    except Exception:
        return name, 'ERR ' + traceback.format_exc()[-1500:]


if __name__ == '__main__':
    flt = sys.argv[1] if len(sys.argv) > 1 else ''
    os.makedirs(OUT, exist_ok=True)
    jobs = [(s, f) for s in SETS for f in sorted(glob.glob(P15 % s + '/*.json')) if flt in os.path.basename(f)]
    print(len(jobs), 'jobs', flush=True)
    with Pool(int(os.environ.get('RULE_POOL', '6')), maxtasksperchild=8) as p:
        for name, st in p.imap_unordered(job, jobs, chunksize=1): print(name, st, flush=True)
