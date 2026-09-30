import json, sys, numpy as np
from collections import Counter
R = json.load(open('/workspace/cl/ideas/r2_rp_rows.json'))
CLEAN = ('hold36', 'prev4')


def tab(rows, title):
    out = '%-52s n=%6d' % (title, len(rows))
    for emb in ['44b6', '6bba', 'clean']:
        x = [r for r in rows if (r['set'] in CLEAN if emb == 'clean' else r['movie'][:4] == emb)]
        g = sum(r['dtp'] == 1 for r in x); l = sum(r['dtp'] == -1 for r in x); dfp = sum(r['dfp'] for r in x)
        ev = sum(1 for r in x if r['dtp'] != 0 or r['dfp'] != 0)
        net = 7.5 * (g - l) - 7.0 * dfp  # units of 1e-6 score
        out += ' | %s +%d -%d dfp%+d net%+.0fu' % (emb, g, l, dfp, net)
    print(out)

E = [r for r in R if r['side'] == 'end' and r['h_a'] >= 2 and not r['q_fdau'] and not r['pc_fork']]
Sd = [r for r in R if r['side'] == 'start']
print('END side (a track>=3, q not fork daughter, pc not fork)')
for lo, hi in [(0, 3.5), (3.5, 5), (5, 7), (7, 9), (9, 12)]:
    x = [r for r in E if lo <= r['d_aq'] < hi]
    tab(x, 'd_aq %.1f-%.1f all' % (lo, hi))
    tab([r for r in x if r['q_start']], '   q START')
    tab([r for r in x if not r['q_start'] and r['h_q'] <= 2], '   q young (1-2 back)')
    tab([r for r in x if r['h_q'] > 2], '   q old (>2 back)')
    tab([r for r in x if r['d_apc'] < r['d_qpc']], '   pc closer to a')
    tab([r for r in x if r['d_apc'] < r['d_qpc'] and r['h_q'] <= 2], '   pc closer to a & q young/start')
    tab([r for r in x if r['fe_new'] > r['fe_old']], '   fe_new > fe_old')
    tab([r for r in x if r['fe_new'] >= 0.5 and r['fe_old'] < 0.5], '   fe_new>=.5 & fe_old<.5')
print('m_a/m_q/m_pc distribution (END, d_aq 3.5-9) with dtp:')
x = [r for r in E if 3.5 <= r['d_aq'] < 9]
print(Counter((r['movie'][:4], r['m_a'], r['m_q'], r['m_pc'], r['dtp'], r['dfp']) for r in x).most_common(25))
print('START side')
for lo, hi in [(0, 3.5), (3.5, 5), (5, 7), (7, 9), (9, 12)]:
    x = [r for r in Sd if lo <= r['d_bc'] < hi]
    tab(x, 'd_bc %.1f-%.1f all' % (lo, hi))
    tab([r for r in x if r['d_pb'] < r['d_pc']], '   b closer to p')
    tab([r for r in x if r['f_b'] > r['f_c']], '   b longer future than c')
    tab([r for r in x if r['fe_new'] > r['fe_old']], '   fe_new > fe_old')
