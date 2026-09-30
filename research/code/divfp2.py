import os, sys, json
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
os.environ['POLARS_MAX_THREADS'] = '2'
import numpy as np
S = np.array([1.625, .40625, .40625])
def job(path):
    import evalx
    import tracking_cellmot.division_metrics as DM
    name = Path(path).stem
    nodes, edges = evalx.load_graph_json(path)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    inv = {v: k for k, v in mapping.items()}
    res = DM.score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    tp = {inv[f] for f in res.tp_forks}; fp = {inv[f] for f in res.fp_forks}
    succ = defaultdict(list); par = {}; eattr = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s; eattr[(s, d)] = e
    forks = [n for n in nodes if len(succ.get(n, [])) >= 2]
    def near_forks(n, k=3):
        found = []
        # ancestors
        c = n; h = 0
        while c in par and h < k:
            c = par[c]; h += 1
            if len(succ.get(c, [])) >= 2: found.append(('anc', h))
        # descendants BFS
        fr = [(n, 0)]
        while fr:
            x, dd = fr.pop()
            if dd >= k: continue
            for y in succ.get(x, []):
                if len(succ.get(y, [])) >= 2: found.append(('desc', dd + 1))
                fr.append((y, dd + 1))
        return found
    out = []
    for n in forks:
        kids = succ[n]
        pos = np.array([nodes[n]['z'], nodes[n]['y'], nodes[n]['x']]) * S
        dk = [float(np.linalg.norm(np.array([nodes[k]['z'], nodes[k]['y'], nodes[k]['x']]) * S - pos)) for k in kids]
        ep = [eattr[(n, k)].get('edge_prob') for k in kids]
        out.append(dict(movie=name, node=n, lab='TP' if n in tp else ('FP' if n in fp else 'unl'), near=len(near_forks(n)), dk_min=min(dk), dk_max=max(dk), ep_min=min([e for e in ep if e is not None], default=None)))
    return out
if __name__ == '__main__':
    import pandas as pd
    files = sorted(Path(sys.argv[1]).glob('*.json'))
    with Pool(len(files)) as pool: res = pool.map(job, [str(f) for f in files], chunksize=1)
    df = pd.DataFrame([r for a in res for r in a])
    print(df.groupby('lab').agg(n=('node', 'size'), near_any=('near', lambda x: (x > 0).mean()), dkmin=('dk_min', 'median'), dkmax=('dk_max', 'median'), epmin=('ep_min', 'median')))
    print(df[df.lab != 'unl'].sort_values(['lab', 'movie']).to_string())
    print('unl forks with near fork:', int((df[df.lab == 'unl'].near > 0).sum()), 'of', int((df.lab == 'unl').sum()))
