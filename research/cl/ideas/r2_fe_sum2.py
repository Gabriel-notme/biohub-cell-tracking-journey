import json, sys, numpy as np
from collections import Counter
R = json.load(open('/workspace/cl/ideas/r2_fe_rows.json'))
P = [r for m in R for r in m['pairs']]
CLEAN = ('hold36', 'prev4')


def tab(rows, title):
    out = '%-58s n=%6d' % (title, len(rows))
    for emb in ['44b6', '6bba', 'clean']:
        x = [r for r in rows if (r['set'] in CLEAN if emb == 'clean' else r['movie'][:4] == emb)]
        c = Counter(r['lab'] for r in x)
        out += ' | %s TP %d FP %d U %d' % (emb, c['TP'], c['FP'], c['U'])
    print(out)

g1 = [r for r in P if r['gap'] == 1]
print('TP pair a_in_full/b_in_full:', Counter((r['a_in_full'], r['b_in_full']) for r in g1 if r['lab'] == 'TP'),
      'FP:', Counter((r['a_in_full'], r['b_in_full']) for r in g1 if r['lab'] == 'FP'))
for H in [5, 10, 20]:
    for dmax in [7, 10, 14, 20]:
        x = [r for r in g1 if r['hist'] >= H and r['fut'] >= H and r['d'] < dmax]
        tab(x, 'g1 hist>=%d fut>=%d d<%d' % (H, H, dmax))
        tab([r for r in x if r['rk_a'] == 0 and r['rk_b'] == 0], '   + MNN')
        tab([r for r in x if r['rk_a'] == 0 and r['rk_b'] == 0 and r['sep_a'] > 3 and r['sep_b'] > 3], '   + MNN sep>3')
        tab([r for r in x if r['rk_a'] == 0 and r['rk_b'] == 0 and 0 <= r['dpred'] < 5], '   + MNN dpred<5')
print('TP pairs detail (d, fe, hist, fut, rk_a, rk_b, dpred, nn_any_t2, b_is_nn):')
for emb in ['44b6', '6bba']:
    x = sorted([r for r in g1 if r['lab'] == 'TP' and r['movie'][:4] == emb], key=lambda r: r['d'])
    print(emb, [(r['d'], round(r['fe'], 2), r['hist'], r['fut'], r['rk_a'], r['rk_b'], r['dpred'], r['nn_any_t2'], r['b_is_nn']) for r in x][:70])
print('TP pair: is b the nearest node of any kind at t+1 to a?', Counter((r['movie'][:4], r['b_is_nn']) for r in g1 if r['lab'] == 'TP'))
print('FP pair: b_is_nn', Counter((r['movie'][:4], r['b_is_nn']) for r in g1 if r['lab'] == 'FP'))
x = [r for r in g1 if r['b_is_nn'] and r['rk_b'] == 0]
for lo, hi in [(0, 4), (4, 7), (7, 10), (10, 14), (14, 20)]:
    tab([r for r in x if lo <= r['d'] < hi], 'g1 b nearest node of a at t+1 & a nearest end of b, d %d-%d' % (lo, hi))
    tab([r for r in x if lo <= r['d'] < hi and r['hist'] >= 3 and r['fut'] >= 3], '   + hist>=3 fut>=3')
