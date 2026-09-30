import os, sys, json, time
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS','RAYON_NUM_THREADS','BLOSC_NTHREADS']: os.environ[k]='1'
sys.path.insert(0, '/workspace/p17ds')
import p17_post, importlib.util
import pandas as pd, numpy as np
from collections import Counter
spec = importlib.util.spec_from_file_location('ps11', '/workspace/p17ds/p_stage11.py'); 
src = open('/workspace/p17ds/p_stage11.py').read()
ns = {}; exec(compile(src.split('def worker')[0], 'ps11', 'exec'), ns); check_graph = ns['check_graph']
def csvload(p):
    df = pd.read_csv(p); out = {}
    for ds, g in df.groupby('dataset'):
        n = g[g.row_type == 'node']; e = g[g.row_type == 'edge']
        pos = {int(r.node_id): (int(r.t), int(r.z), int(r.y), int(r.x)) for r in n.itertuples()}
        out[ds] = (set(pos.values()), set((pos[int(a)], pos[int(b)]) for a, b in zip(e.source_id, e.target_id)))
    return out
def gsets(nodes, edges, f):
    pos = {int(k): (int(v['t']), f(v['z']), f(v['y']), f(v['x'])) for k, v in nodes.items()}
    return set(pos.values()), set((pos[int(e['source_id'])], pos[int(e['target_id'])]) for e in edges)
R = [('round', lambda x: int(round(float(x)))), ('floor+.5', lambda x: int(np.floor(float(x) + .5))), ('trunc', lambda x: int(float(x)))]
for label, gdir, refdir, csv15, csv17, g17dir in [
        ('KAGGLE', '/workspace/kout/p15/pstage_graphs', '/workspace/kout/p15/reference_graphs', '/workspace/kout/p15/submission.csv', '/workspace/kout/p17/submission.csv', None),
        ('LOCAL', '/workspace/cl/ps_p15_prev4/graphs', '/workspace/runs/b5f_prev4/working/reference_graphs', '/workspace/cl/ps_p15_prev4/submission.csv', '/workspace/cl/p16/ps_p17_prev4/submission.csv', '/workspace/cl/p16/ps_p17_prev4/graphs')]:
    C15 = csvload(csv15); C17 = csvload(csv17)
    for m in sorted(C17):
        d = json.load(open(f'{gdir}/{m}.json')); nodes = {int(k): v for k, v in d['nodes'].items()}; edges = d['edges']
        t0 = time.time(); n2, e2, st = p17_post.apply(nodes, edges, f'{refdir}/{m}.json', cutdup_on=1, forkfrag_on=1, shortbranch_on=1); t1 = time.time()
        check_graph(n2, e2, nodes); t2 = time.time()
        res = {}
        for rn, f in R:
            N15, E15 = gsets(nodes, edges, f); N17, E17 = gsets(n2, e2, f)
            res[rn] = dict(p15_match=(N15 == C15[m][0] and E15 == C15[m][1]), p17_match=(N17 == C17[m][0] and E17 == C17[m][1]),
                           n17_diff=(len(N17 ^ C17[m][0]), len(E17 ^ C17[m][1])))
        extra = ''
        if g17dir:
            g = json.load(open(f'{g17dir}/{m}.json')); gn = {int(k) for k in g['nodes']}; ge = {(int(e['source_id']), int(e['target_id'])) for e in g['edges']}
            extra = 'graphdir_exact=%s' % (gn == set(n2) and ge == {(int(e['source_id']), int(e['target_id'])) for e in e2})
        print(label, m, 'nodes %d->%d edges %d->%d' % (len(nodes), len(n2), len(edges), len(e2)), st, 't_apply %.2fs t_check %.2fs' % (t1 - t0, t2 - t1), json.dumps(res), extra, flush=True)
