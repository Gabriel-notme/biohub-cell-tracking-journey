import json
from collections import defaultdict, Counter
import numpy as np
R = json.load(open('/workspace/cl/ideas/r3_tbx4_stress_rows.json'))
VN = ['p0.5', 'p0.5+join', 'p0.8']


def tab(recs, key, bins=None, label=''):
    c = defaultdict(Counter)
    for r in recs:
        v = r[key]
        if bins is not None:
            v = next((b for b in bins if r[key] < b), 'inf')
        c[v][r['res']] += 1; c[v]['n'] += 1; c[v]['nm'] += r['new_matched']
    print('  by %s%s:' % (key, label))
    for k in sorted(c, key=lambda x: (isinstance(x, str), x)):
        z = c[k]; ev = z['TP'] + z['FP']
        print('    %-8s n %6d TP %4d FP %4d UN %6d prec %s  newnode_matchedGT %.3f' % (k, z['n'], z['TP'], z['FP'], z['UN'] + z['X'],
                                                                                 ('%.3f' % (z['TP'] / ev)) if ev else '  -  ', z['nm'] / max(1, z['n'])))


for vi, vn in enumerate(VN):
    print('=== variant', vn)
    for emb in ['44b6', '6bba']:
        recs = [r for m in R if m['movie'].startswith(emb) for r in m['recs'] if r['vi'] == vi]
        c = Counter(r['res'] for r in recs)
        d = [x for m in R if m['movie'].startswith(emb) for x in m['dup'] if x['vi'] == vi]
        print(' %s: added edges %d TP %d FP %d UN %d X %d | added nodes %d matchedGT %d | near-dup pairs among added (<3.5um) %d' % (
            emb, len(recs), c['TP'], c['FP'], c['UN'], c['X'], sum(x['n_add'] for x in d), sum(x['add_matched'] for x in d), sum(x['dup_pairs'] for x in d)))
    recs = [r for m in R for r in m['recs'] if r['vi'] == vi]
    tab(recs, 'kind'); tab(recs, 'side')
    tab(recs, 'prob', [0.6, 0.7, 0.8, 0.9, 0.95, 1.01])
    tab(recs, 'depth', [2, 3, 5, 10, 20, 1000])
    tab(recs, 'bnd')
    tab(recs, 't', [3, 10, 50, 90, 97, 1000])
    tab(recs, 'fdens', [0.8, 0.9, 1.0, 1.1, 1.2, 100])
    tab(recs, 'ldens', [1, 2, 4, 8, 1000])
# dense prev4 movie
for m in R:
    if m['movie'] == '6bba_05db0fb1':
        for vi, vn in enumerate(VN):
            rr = [r for r in m['recs'] if r['vi'] == vi]
            print('6bba_05db0fb1', vn, Counter(r['res'] for r in rr), [x for x in m['dup'] if x['vi'] == vi],
                  'FP recs', [(r['prob'], r['depth'], r['t'], round(r['fdens'], 2), r['ldens'], r['side']) for r in rr if r['res'] == 'FP'])
