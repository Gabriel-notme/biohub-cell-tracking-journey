import os, sys, json, glob
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
os.environ['POLARS_MAX_THREADS'] = '2'
def job(p):
    import evalx
    import tracking_cellmot.division_metrics as DM
    name = Path(p).stem
    nodes, edges = evalx.load_graph_json(p)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    sd = DM.score_divisions(pred, evalx.load_gt(name)[0], scale=evalx.SCALE, max_distance=7.)
    tp = {inv[x] for x in sd.tp_forks}; fp = {inv[x] for x in sd.fp_forks}
    succ = defaultdict(list); par = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s
    def fut(n, lim=10):
        L = 0
        while L < lim and len(succ.get(n, [])) == 1: n = succ[n][0]; L += 1
        return L if len(succ.get(n, [])) == 0 else lim
    def hist(n, lim=10):
        L = 0
        while L < lim and n in par: n = par[n]; L += 1
        return L
    out = []
    for n in nodes:
        if len(succ.get(n, [])) < 2: continue
        f = sorted(fut(c) for c in succ[n])
        out.append(dict(lab='TP' if n in tp else ('FP' if n in fp else 'unl'), fmin=f[0], fmax=f[-1], h=hist(n), added=any(e.get('div_complete') or e.get('dsr') for e in edges if int(e['source_id']) == n)))
    return out
if __name__ == '__main__':
    import pandas as pd
    dfs = []
    for d in sys.argv[1:]:
        with Pool(32) as pool: r = [x for rr in pool.map(job, sorted(glob.glob(d + '/*.json'))) for x in rr]
        df = pd.DataFrame(r); df['set'] = d.split('/')[-2]; dfs.append(df)
    df = pd.concat(dfs)
    print(df.groupby(['set', 'lab']).size().unstack(fill_value=0))
    df['fmin_b'] = pd.cut(df.fmin, [-1, 0, 1, 2, 4, 9, 10])
    print(df[df.lab != 'unl'].groupby(['lab', 'fmin_b']).size().unstack(fill_value=0))
    print(df.groupby(['lab', 'fmin_b']).size().unstack(fill_value=0).loc[['unl']])
    df['h_b'] = pd.cut(df.h, [-1, 0, 1, 2, 4, 9, 10])
    print(df[df.lab != 'unl'].groupby(['lab', 'h_b']).size().unstack(fill_value=0))
