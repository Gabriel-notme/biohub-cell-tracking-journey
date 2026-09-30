"""check2/holdout (c): in-context leave-one-item-out for every touching item of the default P19-R: final graph vs final graph with
that item restored (all rules are deletion-only, so every graph is the P15 graph induced on its node set)."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl/p16/check2/holdout')
import hgrid
from hgrid import sub, step, ORDER, score_detail, B5
from collections import defaultdict
import numpy as np

items = json.load(open('/workspace/cl/p16/check2/holdout/rows1_touch_items.json'))
by = defaultdict(list)
for it in items: by[(it['movie'], it['set'])].append(it)
cfg = json.load(open('/workspace/cl/p16/check2/holdout/cfgs1.json'))[0]
out = []
import evalx
for (movie, s), its in sorted(by.items()):
    refp = B5[s] + '/working/reference_graphs/%s.json' % movie
    n0, e0 = evalx.load_graph_json('/workspace/cl/ps_p15_%s/graphs/%s.json' % (s, movie))
    nodes, edges = n0, e0
    for fam in ORDER: nodes, edges = step(fam, nodes, edges, refp, cfg)
    fin = set(nodes)
    gt, _ = evalx.load_gt(movie)
    rf = evalx.score_movie(movie, nodes, edges)
    tpF, fpF, p2gF = score_detail(nodes, edges, gt)
    g2pF = {g: p for p, g in p2gF.items()}
    for it in its:
        keep = fin | set(it['nodes'])
        nr, er = sub(n0, e0, set(n0) - keep)
        rr = evalx.score_movie(movie, nr, er)
        tpR, fpR, p2gR = score_detail(nr, er, gt)
        ctx = {q: rf[q] - rr[q] for q in ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'num_pred_nodes']}
        # GT nodes matched to the item's nodes when the item is present: who holds them in the final graph?
        held = {str(u): g2pF.get(p2gR[u]) for u in it['nodes'] if u in p2gR}
        it2 = dict(it); it2['ctx'] = ctx; it2['gt_in_restored'] = {str(u): p2gR[u] for u in it['nodes'] if u in p2gR}; it2['gt_holder_final'] = held
        it2['tp_lost_ctx'] = sorted(tpR - tpF); it2['fp_removed_ctx'] = sorted(fpR - fpF); it2['tp_gained_ctx'] = sorted(tpF - tpR); it2['fp_added_ctx'] = sorted(fpF - fpR)
        out.append(it2)
        print('%-14s %-3s n=%d ctx tp %+d fp %+d fn %+d | P15-marg tp %+d fp %+d' % (movie, it['fam'], len(it['nodes']), ctx['edge_tp'], ctx['edge_fp'], ctx['edge_fn'],
                                                                            it['marg']['edge_tp'], it['marg']['edge_fp']), flush=True)
json.dump(out, open('/workspace/cl/p16/check2/holdout/ctx_items.json', 'w'), default=str)
