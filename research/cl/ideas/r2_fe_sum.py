import json, sys, numpy as np
from collections import Counter
R = json.load(open('/workspace/cl/ideas/r2_fe_rows.json'))
E = [r for m in R for r in m['ends']]; St = [r for m in R for r in m['starts']]; P = [r for m in R for r in m['pairs']]
q = lambda v: np.round(np.quantile(np.array(v, float), [.1, .25, .5, .75, .9]), 2).tolist() if len(v) else []
print('=== END anatomy (free ends, t<tmax) ===')
for emb in ['44b6', '6bba']:
    x = [r for r in E if r['movie'][:4] == emb]
    print(emb, 'ends', len(x), Counter(r['cls'] for r in x))
    print('   gt_cont sub', Counter((r.get('sub'), r.get('sub2')) for r in x if r['cls'] == 'gt_cont'))
    print('   fork-daughter ends', Counter(r['cls'] for r in x if r['fdau']))
    y = [r for r in x if r.get('sub') == 'child_start']
    print('   child_start: dist', q([r['dc'] for r in y]), 'fe', Counter('nocand' if r['fe_c'] < 0 else ('>=.5' if r['fe_c'] >= .5 else ('.3-.5' if r['fe_c'] >= .3 else '<.3')) for r in y))
    y = [r for r in x if r.get('sub') == 'child_held']
    print('   child_held: dist', q([r['dc'] for r in y]), 'holder-parent dist to end', q([r['held_dpar'] for r in y]), Counter(r['held_pmatch'] for r in y))
print('=== START anatomy ===')
for emb in ['44b6', '6bba']:
    x = [r for r in St if r['movie'][:4] == emb]
    print(emb, 'starts', len(x), Counter(r['cls'] for r in x), 'gt_cont sub', Counter(r.get('sub') for r in x if r['cls'] == 'gt_cont'))


def tab(rows, title):
    c = Counter((r['movie'][:4], r['lab']) for r in rows)
    out = title + ' n=%d' % len(rows)
    for emb in ['44b6', '6bba']:
        tp, fp, u = c[(emb, 'TP')], c[(emb, 'FP')], c[(emb, 'U')]
        out += ' | %s TP %d FP %d U %d prec %.2f' % (emb, tp, fp, u, tp / max(1, tp + fp))
    print(out)

print('=== PAIRS gap1 by distance bin, fe band ===')
g1 = [r for r in P if r['gap'] == 1]
for lo, hi in [(0, 4), (4, 7), (7, 10), (10, 14), (14, 20), (20, 30)]:
    for band, f in [('nocand', lambda r: r['fe'] < 0), ('fe<.1', lambda r: 0 <= r['fe'] < .1), ('fe.1-.3', lambda r: .1 <= r['fe'] < .3),
                    ('fe.3-.5', lambda r: .3 <= r['fe'] < .5), ('fe>=.5', lambda r: r['fe'] >= .5)]:
        x = [r for r in g1 if lo <= r['d'] < hi and f(r)]
        if x: tab(x, 'g1 %2d-%2d %-8s' % (lo, hi, band))
print('=== PAIRS gap1 mutual-nearest (rk_a==0 and rk_b==0) ===')
for lo, hi in [(0, 7), (7, 10), (10, 14), (14, 20), (20, 30)]:
    for band, f in [('nocand', lambda r: r['fe'] < 0), ('fe<.3', lambda r: 0 <= r['fe'] < .3), ('fe.3-.5', lambda r: .3 <= r['fe'] < .5), ('fe>=.5', lambda r: r['fe'] >= .5)]:
        x = [r for r in g1 if lo <= r['d'] < hi and f(r) and r['rk_a'] == 0 and r['rk_b'] == 0]
        if x: tab(x, 'MNN g1 %2d-%2d %-8s' % (lo, hi, band))
print('=== PAIRS gap2 by distance ===')
g2 = [r for r in P if r['gap'] == 2]
for lo, hi in [(0, 4), (4, 7), (7, 10), (10, 14), (14, 18), (18, 30)]:
    x = [r for r in g2 if lo <= r['d'] < hi]
    if x: tab(x, 'g2 %2d-%2d all' % (lo, hi))
    x = [r for r in g2 if lo <= r['d'] < hi and r['rk_a'] == 0 and r['rk_b'] == 0]
    if x: tab(x, 'g2 %2d-%2d MNN' % (lo, hi))
