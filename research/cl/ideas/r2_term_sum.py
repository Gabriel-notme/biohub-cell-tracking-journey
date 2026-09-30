import json, numpy as np
from collections import Counter
R = json.load(open('/workspace/cl/ideas/r2_term_rows.json'))
CLEAN = ('hold36', 'prev4')


def tab(rows, title):
    out = '%-44s n=%6d' % (title, len(rows))
    for emb in ['44b6', '6bba', 'clean']:
        x = [r for r in rows if (r['set'] in CLEAN if emb == 'clean' else r['movie'][:4] == emb)]
        c = Counter(r['lab'] for r in x)
        out += ' | %s TP %d FP %d U %d fp%% %.2f' % (emb, c['TP'], c['FP'], c['U'], c['FP'] / max(1, c['TP'] + c['FP']))
    print(out)

for side in ['end', 'start']:
    X = [r for r in R if r['side'] == side]
    print('=====', side)
    tab(X, 'all terminal edges')
    for lo, hi in [(0, 4), (4, 7), (7, 10), (10, 14), (14, 40)]:
        tab([r for r in X if lo <= r['step'] < hi], 'step %d-%d' % (lo, hi))
    for lo, hi in [(0, 3), (3, 6), (6, 9), (9, 40)]:
        tab([r for r in X if lo <= r['res'] < hi], 'motion residual %d-%d' % (lo, hi))
    for lo, hi in [(0, 3.5), (3.5, 5), (5, 7), (7, 10), (10, 99)]:
        tab([r for r in X if lo <= r['dcont'] < hi], 'dist to nearest continuing node %.1f-%.1f' % (lo, hi))
    for h in [(1, 1), (2, 2), (3, 5), (6, 50)]:
        tab([r for r in X if h[0] <= r['hist'] <= h[1]], 'track len-1 %d-%d' % h)
    fl = Counter(r['flag'] for r in X)
    for f, n in fl.most_common(8):
        tab([r for r in X if r['flag'] == f], 'flag [%s]' % f[:36])
    tab([r for r in X if r['res'] >= 6 and r['step'] >= 7], 'res>=6 & step>=7')
    tab([r for r in X if r['res'] >= 6 and r['dcont'] < 7], 'res>=6 & dcont<7')
    tab([r for r in X if r['fe'] < 0.1], 'fe<0.1')
