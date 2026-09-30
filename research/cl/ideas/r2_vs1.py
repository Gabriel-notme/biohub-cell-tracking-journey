"""Re-score a rule_eval dump against variant 1 (P14) instead of variant 0 (P13). usage: r2_vs1.py <dump.json>"""
import sys, json, numpy as np
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
R = json.load(open(sys.argv[1]))
V = sorted(set(r['vi'] for r in R))
base = {r['movie']: r for r in R if r['vi'] == 1}
ms = sorted(base); rng = np.random.default_rng(0); K = [rng.integers(0, len(ms), len(ms)) for _ in range(300)]
for i in V:
    if i <= 1: continue
    cur = {r['movie']: r for r in R if r['vi'] == i}
    line = 'v%d' % i
    for emb in ['44b6', '6bba', '']:
        mm = [m for m in ms if m.startswith(emb)]
        a = summarise([base[m] for m in mm]); b = summarise([cur[m] for m in mm])
        line += ' | %s %+.5f (e %+.5f, div %+d/%+d)' % (emb or 'all', b['score'] - a['score'], b['adj_edge_jaccard'] - a['adj_edge_jaccard'],
                                                     b['division_tp'] - a['division_tp'], b['division_fp'] - a['division_fp'])
    bs = [summarise([cur[ms[j]] for j in k])['score'] - summarise([base[ms[j]] for j in k])['score'] for k in K]
    line += ' | CI [%+.5f, %+.5f]' % (np.quantile(bs, .025), np.quantile(bs, .975))
    cm = [m for m in ms if base[m]['set'] in ('hold36', 'prev4')]
    line += ' | clean40 %+.5f' % (summarise([cur[m] for m in cm])['score'] - summarise([base[m] for m in cm])['score'])
    stats = {}
    for k in ('ll_added', 'mnn_added'):
        v = sum(cur[m].get(k, 0) or 0 for m in ms)
        if v: stats[k] = v
    print(line, stats, flush=True)
