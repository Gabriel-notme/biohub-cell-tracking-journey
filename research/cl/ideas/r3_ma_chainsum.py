"""Summarise r3_ma_chainlab_rows.json: per chain class, TP / FP / nodes and the linearised net value on the 199-movie score
(TP: +1/W, FP: -J/W, node: -0.1*J_i*w_i/(Nt_i*W)), per embryo and on clean40."""
import json
from collections import defaultdict
R = json.load(open('/workspace/cl/ideas/r3_ma_chainlab_rows.json'))
W = sum(r['w0'] for r in R)
print('W', W, 'collateral', {k: sum(r['coll'].get(k, 0) for r in R) for k in set(k for r in R for k in r['coll'])})
print('total d', {k: sum(r['d'][k] for r in R) for k in ['tp', 'fp', 'fn', 'nodes']})


def val(rows):
    out = {}
    for emb in ['44b6', '6bba', 'clean']:
        tp = fp = nd = 0; v = 0.
        for r, x in rows:
            if emb == 'clean':
                if r['set'] not in ('hold36', 'prev4'): continue
            elif r['emb'] != emb: continue
            tp += x['tp']; fp += x['fp']; nd += x['L']
            v += (x['tp'] - r['J0'] * x['fp']) / W - 0.1 * r['J0'] * r['w0'] / (r['n_total'] * W) * x['L']
        out[emb] = (tp, fp, nd, v)
    return out


ALL = [(r, x) for r in R for x in r['rows']]
F = {
    'all': lambda x: True,
    'L>=3': lambda x: x['L'] >= 3,
    'L>=4': lambda x: x['L'] >= 4,
    'L=5': lambda x: x['L'] == 5,
    'pmin>=0.8': lambda x: x['pmin'] >= 0.8,
    'dse<=10': lambda x: x['dse'] <= 10,
    'dse<=6': lambda x: x['dse'] <= 6,
    'dse>10': lambda x: x['dse'] > 10,
    'dkmin>=5': lambda x: x['dkmin'] >= 5,
    'dkmin>=7': lambda x: x['dkmin'] >= 7,
    'bxy>=4&bz>=3': lambda x: x['bxy'] >= 4 and x['bz'] >= 3,
    'side s': lambda x: x['side'] == 's',
    'side e': lambda x: x['side'] == 'e',
    'L>=3&dkmin>=5': lambda x: x['L'] >= 3 and x['dkmin'] >= 5,
    'L>=4&dkmin>=5': lambda x: x['L'] >= 4 and x['dkmin'] >= 5,
    'L>=3&pmin>=0.8&dkmin>=5': lambda x: x['L'] >= 3 and x['pmin'] >= 0.8 and x['dkmin'] >= 5,
}
print('%-26s | %-34s | %-34s | %-34s' % ('filter', '44b6 tp/fp/nodes net', '6bba tp/fp/nodes net', 'clean40 tp/fp/nodes net'))
for k, f in F.items():
    o = val([(r, x) for r, x in ALL if f(x)])
    print('%-26s | ' % k + ' | '.join('%4d/%3d/%6d %+.5f' % o[e] for e in ['44b6', '6bba', 'clean']))
print('\nby L:')
for L in [2, 3, 4, 5, 6, 7]:
    o = val([(r, x) for r, x in ALL if x['L'] == L])
    print('L=%d  ' % L + ' | '.join('%4d/%3d/%6d %+.5f' % o[e] for e in ['44b6', '6bba', 'clean']))
print('\nby dkmin (nearest kept node in own frame, um):')
for lo, hi in [(3.5, 5), (5, 7), (7, 10), (10, 99)]:
    o = val([(r, x) for r, x in ALL if lo <= x['dkmin'] < hi])
    print('%4.1f-%4.1f ' % (lo, hi) + ' | '.join('%4d/%3d/%6d %+.5f' % o[e] for e in ['44b6', '6bba', 'clean']))
