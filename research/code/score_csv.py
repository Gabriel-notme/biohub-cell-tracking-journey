import sys, json
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
import pandas as pd, evalx
from tracking_cellmot.metrics import summarise
rows = []
for path in sys.argv[1:]:
    df = pd.read_csv(path); res = []
    for ds, g in df.groupby('dataset'):
        n = g[g.row_type == 'node']; e = g[g.row_type == 'edge']
        nodes = {int(r.node_id): {'t': int(r.t), 'z': float(r.z), 'y': float(r.y), 'x': float(r.x)} for r in n.itertuples()}
        edges = [{'source_id': int(s), 'target_id': int(t)} for s, t in zip(e.source_id, e.target_id)]
        r = evalx.score_movie(ds, nodes, edges); res.append(r)
        print(path.split('/')[-2], ds, 'tp %d fp %d fn %d div %d/%d/%d npred %d adj %.4f' % (r['edge_tp'], r['edge_fp'], r['edge_fn'], r['division_tp'], r['division_fp'], r['division_fn'], r['num_pred_nodes'], r['adj_edge_jaccard']))
    s = summarise(res); print('SUMMARY', path, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in s.items()}))
