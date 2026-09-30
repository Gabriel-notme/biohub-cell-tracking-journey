"""chk2 placebo (info): observed sequential increments and leave-one-family-out of P19-R on P15 (exact scorer).
usage: loo.py  -> rows_loo.json ; then loo.py agg"""
import os, sys, json, glob, traceback
sys.path.insert(0, '/workspace/cl/p16/check2/placebo')
from lib import *
from multiprocessing import Pool
KEYS = ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes']
OUT = '/workspace/cl/p16/check2/placebo/rows_loo.json'


def job(args):
    s, f = args
    name = os.path.basename(f)[:-5]
    try:
        n15, e15 = load(f); refp = B5[s] + '/working/reference_graphs/%s.json' % name
        steps, (nf, ef), D = apply_rules(n15, e15, refp)
        rm = {fam: r for fam, _, _, r in steps}; allr = set().union(*rm.values())
        sc = Scorer(name, n15, e15)
        V = {'p15': set(), 'P19': allr}
        cum = set()
        for fam in FAMS: cum = cum | rm[fam]; V['seq_' + fam] = set(cum)
        for fam in FAMS: V['loo_' + fam] = allr - rm[fam]
        V['loo_dup'] = allr - rm['st'] - rm['tt'] - rm['par']
        V['loo_tt'] = allr - rm['tt']
        out = {}
        for k, drop in V.items():
            r = sc.score(drop); out[k] = [r[x] for x in KEYS[:6]] + [len(n15) - len(drop)]
        return {'movie': name, 'set': s, 'n_total': sc.nt, 'rows': out}
    except Exception:
        return {'movie': name, 'err': traceback.format_exc()[-1500:]}


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'agg':
        import numpy as np
        sys.path.insert(0, '/workspace/official/src')
        from tracking_cellmot.metrics import summarise, per_sample_metrics, EvaluationResult
        L = json.load(open(OUT)); L = [x for x in L if 'rows' in x]; ms = [x['movie'] for x in L]
        print('movies', len(L))

        def rows(v, sel): return [per_sample_metrics(EvaluationResult(*x['rows'][v]), x['n_total'], 0.0) for x in L if sel(x)]
        G = [('all', lambda x: True), ('44b6', lambda x: x['movie'].startswith('44b6')), ('6bba', lambda x: x['movie'].startswith('6bba')),
             ('clean40', lambda x: x['set'] in ('hold36', 'prev4'))]
        rng = np.random.default_rng(0); idx = [rng.integers(0, len(L), len(L)) for _ in range(1000)]
        for v in ['P19'] + ['seq_' + f for f in FAMS] + ['loo_' + f for f in FAMS] + ['loo_dup']:
            prev = {'seq_cd': 'p15', 'seq_ff': 'seq_cd', 'seq_st': 'seq_ff', 'seq_tt': 'seq_st', 'seq_par': 'seq_tt', 'seq_bd': 'seq_par'}.get(v, 'p15')
            ref = 'P19' if v.startswith('loo_') else prev
            line = '%-9s vs %-8s' % (v, ref)
            for g, sel in G:
                a = summarise(rows(ref, sel))['score']; b = summarise(rows(v, sel))['score']; line += ' | %s %+.6f' % (g, b - a)
            bs = []
            for k in idx:
                sub_ = [L[i] for i in k]
                ra = [per_sample_metrics(EvaluationResult(*x['rows'][ref]), x['n_total'], 0.0) for x in sub_]
                rb = [per_sample_metrics(EvaluationResult(*x['rows'][v]), x['n_total'], 0.0) for x in sub_]
                bs.append(summarise(rb)['score'] - summarise(ra)['score'])
            line += ' | CI [%+.6f, %+.6f]' % (np.quantile(bs, .025), np.quantile(bs, .975))
            dc = [sum(x['rows'][v][i] - x['rows'][ref][i] for x in L) for i in range(7)]
            line += ' | dTP %+d dFP %+d dFN %+d div %+d/%+d nodes %+d' % (dc[0], dc[1], dc[2], dc[3], dc[4], dc[6])
            print(line, flush=True)
        sys.exit()
    jobs = [(s, f) for s in SETS for f in sorted(glob.glob(P15 % s + '/*.json'))]
    with Pool(int(os.environ.get('RULE_POOL', '6')), maxtasksperchild=8) as p: res = p.map(job, jobs, chunksize=1)
    print('errors', sum('err' in r for r in res))
    json.dump(res, open(OUT, 'w'))
