"""Why are preview-4 scores low? Decompose per movie and per set: edge J, node multiplier, division term, GT size, density."""
import json, sys
import numpy as np
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
for cfg in ['b5', 'p6n', 'p9']:
    for s in ['hold36', 'prev4', 'audit32']:
        try: rows = json.load(open('/workspace/cl/rev/%s_%s.json' % (cfg, s)))
        except Exception: continue
        S = summarise(rows)
        tp = sum(r['edge_tp'] for r in rows); fp = sum(r['edge_fp'] for r in rows); fn = sum(r['edge_fn'] for r in rows)
        mult = np.average([1 - 0.1 * r['total_node_ratio'] for r in rows], weights=[r['edge_tp'] + r['edge_fp'] + r['edge_fn'] for r in rows])
        dtp, dfp, dfn = S['division_tp'], S['division_fp'], S['division_fn']
        print('%-5s %-8s score %.4f = adjE %.4f + 0.1*divJ %.4f | edgeJ %.4f  mult %.4f | GT edges/movie %.0f  pred nodes/movie %.0f  ntot/movie %.0f | div %d/%d/%d' % (
            cfg, s, S['score'], S['adj_edge_jaccard'], 0.1 * S['division_jaccard'], tp / (tp + fp + fn), mult, (tp + fn) / len(rows),
            np.mean([r['num_pred_nodes'] for r in rows]), np.mean([r['n_total'] for r in rows]), dtp, dfp, dfn))
print()
for r in json.load(open('/workspace/cl/rev/p9_prev4.json')):
    print('prev4 movie %s edgeJ %.4f adjE %.4f TP/FP/FN %d/%d/%d nodes %d ntot %.0f ratio %+.3f div %d/%d/%d' % (
        r['movie'], r['edge_jaccard'], r['adj_edge_jaccard'], r['edge_tp'], r['edge_fp'], r['edge_fn'], r['num_pred_nodes'], r['n_total'], r['total_node_ratio'],
        r['division_tp'], r['division_fp'], r['division_fn']))
