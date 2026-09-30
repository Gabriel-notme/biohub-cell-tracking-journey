"""Read-only analysis: double forks remaining in P13 final graphs (after relink/edge_link, which run after dfork),
fork origins and TP/FP labels. usage: python3 ideas/pp_dd_an.py"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool

SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']


def origin(e):
    for k in ['div_complete', 'relink', 'edge_link']:
        if k in e: return k
    return 'b5'


def job(f):
    import evalx
    import tracking_cellmot.division_metrics as DM
    name = Path(f).stem; st = f.split('ps_p13_')[1].split('/')[0]
    nodes, edges = evalx.load_graph_json(f)
    succ = defaultdict(list); par = {}; eo = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s; eo[(s, d)] = origin(e)
    forks = [n for n in nodes if len(succ.get(n, [])) >= 2]
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    sd = DM.score_divisions(pred, evalx.load_gt(name)[0], scale=evalx.SCALE, max_distance=7.)
    tp = {inv[x] for x in sd.tp_forks}; fpf = {inv[x] for x in sd.fp_forks}
    lab = lambda n: 'TP' if n in tp else ('FP' if n in fpf else 'unl')
    c = Counter(); recs = []
    for f0 in forks:
        c['fork_' + lab(f0)] += 1
        c['fork_origin_%s_%s' % ('+'.join(sorted({eo[(f0, x)] for x in succ[f0]})), lab(f0))] += 1
        # ancestor forks (whole movie) along the parent chain
        x = f0; path_orig = set(); dt = 0
        while x in par:
            p = par[x]; path_orig.add(eo[(p, x)]); dt += 1
            if len(succ.get(p, [])) >= 2:
                recs.append({'movie': name, 'set': st, 'anc': p, 'desc': f0, 'dt': dt, 'lab_anc': lab(p), 'lab_desc': lab(f0),
                             'path_orig': sorted(path_orig)})
                break
            x = p
    return c, recs


if __name__ == '__main__':
    fs = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(48) as p: R = p.map(job, fs, chunksize=1)
    C = Counter()
    for c, _ in R: C.update(c)
    print(dict(sorted(C.items())))
    recs = [r for _, rs in R for r in rs]
    print('ancestor-descendant fork pairs', len(recs))
    cc = Counter((r['dt'] <= 35, tuple(r['path_orig']), r['lab_anc'], r['lab_desc'], r['movie'][:4]) for r in recs)
    for k, v in sorted(cc.items(), key=lambda kv: -kv[1]): print(k, v)
    json.dump(recs, open('/workspace/cl/ideas/pp_dd_recs.json', 'w'))
