"""Reviewer check (read-only): label P13 div_complete forks (official DM.score_divisions), tabulate by flags on the continuation
edge p->a, per embryo / clean. Verifies dl_jeveto claims and shows the multiplicity of flag categories."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
SKIP = {'source_id', 'target_id', 'edge_prob', 'edge_dist', 'dist', 'prob', 'score', 'cost'}


def job(a):
    s, f = a
    import evalx
    import tracking_cellmot.division_metrics as DM
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    res = DM.score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    tpf = {inv[x] for x in res.tp_forks}; fpf = {inv[x] for x in res.fp_forks}
    ch = defaultdict(list); E = {}
    for e in edges:
        u, v = int(e['source_id']), int(e['target_id']); ch[u].append(v); E[(u, v)] = e
    out = []
    for p, ks in ch.items():
        if len(ks) != 2: continue
        dcs = [k for k in ks if 'div_complete' in E[(p, k)]]
        lab = 'TP' if p in tpf else ('FP' if p in fpf else 'U')
        if len(dcs) != 1:
            out.append({'lab': lab, 'dc': 0, 'flags': [], 'mv': name}); continue
        b = dcs[0]; a = [k for k in ks if k != b][0]
        fl = sorted(k for k, v in E[(p, a)].items() if k not in SKIP and not isinstance(v, float))
        allk = sorted(k for k in E[(p, a)] if k not in ('source_id', 'target_id'))
        out.append({'lab': lab, 'dc': 1, 'flags': fl, 'allk': allk, 'mv': name})
    return s, name, out


if __name__ == '__main__':
    jobs = [(s, f) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(24) as p: R = p.map(job, jobs, chunksize=1)
    json.dump(R, open('/workspace/cl/ideas/rv_jev_chk_rows.json', 'w'))
    kc = Counter()
    for s, n, out in R:
        for r in out:
            if r['dc']: kc.update(r['allk'])
    print('continuation-edge key counts (dc forks):', kc.most_common())
    def tab(name, fn):
        C = defaultdict(Counter)
        for s, n, out in R:
            e = n[:4]; cl = s in ('hold36', 'prev4')
            for r in out:
                if not r['dc']: continue
                k = fn(r); C[k][(e, r['lab'])] += 1
                if cl: C[k][('clean', r['lab'])] += 1
        print('==', name)
        for k in sorted(C, key=str):
            c = C[k]; print('  %-40s' % str(k)[:40], ' | '.join('%s TP %2d FP %2d U %4d' % (g, c[(g, 'TP')], c[(g, 'FP')], c[(g, 'U')]) for g in ['44b6', '6bba', 'clean']))
    tab('all dc forks', lambda r: 'all')
    for k in sorted(kc):
        tab('has ' + k, lambda r, k=k: k in r['allk'])
    tab('joint_event|learned_recovery', lambda r: ('joint_event' in r['allk']) or ('learned_recovery' in r['allk']))
    for s, n, out in R:
        for r in out:
            if r['dc'] and r['lab'] == 'FP' and (('joint_event' in r['allk']) or ('learned_recovery' in r['allk'])):
                print('JEV FP', s, n, r['allk'])
    tot = Counter((n[:4], r['lab']) for s, n, out in R for r in out)
    print('all forks', sorted(tot.items()))
