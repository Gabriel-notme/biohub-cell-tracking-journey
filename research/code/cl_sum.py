import json, glob, sys
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
for s in ['hold36', 'audit32', 'prev4']:
    for c in ['b5', 'p3', 'p4']:
        f = '/workspace/pool/%s__%s.json' % (s, c)
        try: rows = json.load(open(f))
        except Exception as e: print(s, c, 'missing'); continue
        sm = summarise(rows)
        tp = sum(r['edge_tp'] for r in rows); fp = sum(r['edge_fp'] for r in rows); fn = sum(r['edge_fn'] for r in rows)
        npred = sum(r['num_pred_nodes'] for r in rows); ntot = sum(r['n_total'] for r in rows)
        ratios = [ (r['num_pred_nodes'] - r['n_total'])/r['n_total'] for r in rows]
        print('%-8s %-4s score %.5f adjE %.5f E %.5f  eTP %d eFP %d eFN %d  div %d/%d/%d  npred/ntot %.3f  ratio min %.3f max %.3f' % (s, c, sm['score'], sm['adj_edge_jaccard'], sm['edge_jaccard'], tp, fp, fn, sm['division_tp'], sm['division_fp'], sm['division_fn'], npred/ntot, min(ratios), max(ratios)))
print(sorted(rows[0].keys()))
