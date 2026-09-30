import os, sys, json
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
os.environ['POLARS_MAX_THREADS'] = '2'
def job(path):
    import evalx
    import tracking_cellmot.division_metrics as DM
    name = Path(path).stem
    nodes, edges = evalx.load_graph_json(path)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    inv = {v: k for k, v in mapping.items()}
    res = DM.score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    ev, cross, mal = DM._pred_division_fork_sets(pred, gt, evalx.SCALE, 7.)
    out = []
    pst = {}
    for f in res.fp_forks:
        cat = 'evaluable' if f in ev else ('cross' if f in cross else ('malformed' if f in mal else 'considered'))
        n = inv[f]
        out.append(dict(movie=name, node=n, cat=cat, t=nodes[n]['t']))
    for f in res.tp_forks: out.append(dict(movie=name, node=inv[f], cat='TP', t=nodes[inv[f]]['t']))
    return out
if __name__ == '__main__':
    import pandas as pd
    files = sorted(Path(sys.argv[1]).glob('*.json'))
    with Pool(len(files)) as pool: res = pool.map(job, [str(f) for f in files], chunksize=1)
    df = pd.DataFrame([r for a in res for r in a])
    # load pstage info on added forks if present
    added = {}
    for f in files:
        g = json.load(open(f)); ps = g.get('pstage') or {}
        for k in ('added_forks', 'forks_added_list'):
            if k in ps: added[f.stem] = set(ps[k])
    print(df.cat.value_counts())
    print(df.to_string())
