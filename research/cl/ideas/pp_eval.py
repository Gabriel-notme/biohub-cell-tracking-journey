"""rule_eval.py clone that also prints summed integer stats per variant, per-set (clean40 per embryo) deltas and div/edge counts.
usage: python3 ideas/pp_eval.py <module> '<json list of kwargs>'"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from multiprocessing import Pool
import numpy as np
import rule_eval as RE

if __name__ == '__main__':
    mod = sys.argv[1]; V = [None] + json.loads(sys.argv[2])
    jobs = [(mod, f, V) for s, d in RE.SRC['p13'].items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(int(os.environ.get('RULE_POOL', '60'))) as p: R = [r for rs in p.map(RE.job, jobs, chunksize=1) for r in rs]
    from tracking_cellmot.metrics import summarise
    rng = np.random.default_rng(0)
    base = {r['movie']: r for r in R if r['vi'] == 0}
    ms = sorted(base); Kb = [rng.integers(0, len(ms), len(ms)) for _ in range(300)]
    skip = {'movie', 'vi', 'set', 'n_total', 'edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes'}
    for i, kw in enumerate(V[1:], 1):
        cur = {r['movie']: r for r in R if r['vi'] == i}
        out = [json.dumps(kw)]
        for name, mm in [('44b6', [m for m in ms if m.startswith('44b6')]), ('6bba', [m for m in ms if m.startswith('6bba')]), ('all', ms),
                         ('c40_44b6', [m for m in ms if base[m]['set'] in ('hold36', 'prev4') and m.startswith('44b6')]),
                         ('c40_6bba', [m for m in ms if base[m]['set'] in ('hold36', 'prev4') and m.startswith('6bba')]),
                         ('clean40', [m for m in ms if base[m]['set'] in ('hold36', 'prev4')])]:
            a = summarise([base[m] for m in mm]); b = summarise([cur[m] for m in mm])
            dd = {k: sum(cur[m][k] - base[m][k] for m in mm) for k in ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'num_pred_nodes']}
            out.append('  %-9s %+.5f (adjE %+.5f) eTP %+d eFP %+d divTP %+d divFP %+d nodes %+d' % (name, b['score'] - a['score'], b['adj_edge_jaccard'] - a['adj_edge_jaccard'],
                       dd['edge_tp'], dd['edge_fp'], dd['division_tp'], dd['division_fp'], dd['num_pred_nodes']))
        bs = [summarise([cur[ms[j]] for j in k])['score'] - summarise([base[ms[j]] for j in k])['score'] for k in Kb]
        out.append('  CI [%+.5f, %+.5f]' % (np.quantile(bs, .025), np.quantile(bs, .975)))
        st = {}
        for m in ms:
            for k, v in cur[m].items():
                if k not in skip and isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ('edge_jaccard', 'adj_edge_jaccard', 'node_recall', 'total_node_ratio'):
                    st[k] = st.get(k, 0) + v
        out.append('  stats ' + json.dumps({k: round(v, 3) for k, v in st.items()}))
        chg = [m for m in ms if abs(cur[m]['edge_tp'] - base[m]['edge_tp']) + abs(cur[m]['edge_fp'] - base[m]['edge_fp']) + abs(cur[m]['division_tp'] - base[m]['division_tp']) + abs(cur[m]['division_fp'] - base[m]['division_fp']) > 0]
        out.append('  movies changed %d' % len(chg))
        print('\n'.join(out), flush=True)
