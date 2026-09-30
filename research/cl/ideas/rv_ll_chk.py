"""Reviewer check (read-only): lengths of pp_longlink additions vs GT edge lengths and B5 edge lengths; whether endpoints were B5 ends/starts."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import rule_eval as RE
S = np.array([1.625, .40625, .40625])
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}

def elen(nodes, a, b):
    return float(np.linalg.norm((np.array([nodes[a][k] for k in 'zyx'], float) - np.array([nodes[b][k] for k in 'zyx'], float)) * S))

def job(f):
    import evalx
    from ideas import pp_longlink
    name = Path(f).stem; st = [k for k in RE.FULL if ('_%s/' % k) in f][0]
    nodes, edges = evalx.load_graph_json(f)
    n2, e2, s2 = pp_longlink.apply(nodes, edges, name=name, set=st, fullgeff=RE.FULL[st] + '/' + name + '.geff', fe_min=0.5)
    add = [e for e in e2 if 'long_link' in e]
    # B5 graph
    bf = glob.glob(B5[st] + '/' + name + '*')
    b5max = None; b5ends = None
    if bf:
        try:
            bn, be = evalx.load_graph_json(bf[0])
            ls = [elen(bn, int(e['source_id']), int(e['target_id'])) for e in be if int(e['source_id']) in bn and int(e['target_id']) in bn]
            b5max = max(ls) if ls else None
            bch = {int(e['source_id']) for e in be}; bpa = {int(e['target_id']) for e in be}
            b5ends = sum(1 for e in add if int(e['source_id']) in bn and int(e['source_id']) not in bch and int(e['target_id']) in bn and int(e['target_id']) not in bpa)
        except Exception as ex:
            b5max = 'err:' + str(ex)[:60]
    p13ls = [elen(nodes, int(e['source_id']), int(e['target_id'])) for e in edges]
    # GT edge lengths
    gtn, gte = evalx.load_gt(name) if False else (None, None)
    return dict(movie=name, set=st, add=[round(elen(n2, int(e['source_id']), int(e['target_id'])), 2) for e in add], b5max=b5max, b5ends=b5ends, n_add=len(add), p13max=max(p13ls) if p13ls else 0)

if __name__ == '__main__':
    fs = [f for s, d in RE.SRC['p13'].items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(48) as p: res = p.map(job, fs, chunksize=1)
    json.dump(res, open('/workspace/cl/ideas/rv_ll_chk.json', 'w'))
    for emb in ['44b6', '6bba']:
        r = [x for x in res if x['movie'].startswith(emb)]
        a = np.array([d for x in r for d in x['add']])
        print(emb, 'movies', len(r), 'added', len(a), 'len q', np.round(np.quantile(a, [0, .5, .9, .99, 1]), 1).tolist() if len(a) else None,
              '>20um', int((a > 20).sum()) if len(a) else 0, '>25um', int((a > 25).sum()) if len(a) else 0,
              'b5max q', np.round(np.quantile([x['b5max'] for x in r if isinstance(x['b5max'], float)], [.5, .9, 1]), 1).tolist(),
              'p13max', round(max(x['p13max'] for x in r), 1),
              'added both B5 end/start', sum(x['b5ends'] or 0 for x in r))
    print('b5 errors', [x['b5max'] for x in res if isinstance(x['b5max'], str)][:3], 'b5 missing', sum(1 for x in res if x['b5max'] is None))
