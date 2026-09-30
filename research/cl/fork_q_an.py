import json, glob
import numpy as np
R = []
for f in glob.glob('/workspace/cl/forkq/*.json'):
    s, m = f.split('/')[-1][:-5].split('__')
    for r in json.load(open(f)): r['set'] = s; r['emb'] = m[:4]; R.append(r)
print('forks', len(R), 'movies', len(glob.glob('/workspace/cl/forkq/*.json')))
filters = {
    'all': lambda r: True,
    'persist5': lambda r: r['la'] >= 5 and r['lb'] >= 5,
    'persist5+sep': lambda r: r['la'] >= 5 and r['lb'] >= 5 and r['d4'] > r['d0'],
    'persist8+sep+hist5': lambda r: r['la'] >= 8 and r['lb'] >= 8 and r['d4'] > r['d0'] and r['lp'] >= 5,
    'p8+sep+h5+d8>d0+2': lambda r: r['la'] >= 8 and r['lb'] >= 8 and r['d8'] > r['d0'] + 2 and r['lp'] >= 5,
}
for emb in ['44b6', '6bba', 'all']:
    for k, fn in filters.items():
        Q = [r for r in R if fn(r) and (emb == 'all' or r['emb'] == emb)]
        T = sum(r['lab'] == 'T' for r in Q); F = sum(r['lab'] == 'F' for r in Q); U = sum(r['lab'] == 'U' for r in Q)
        print('%-4s %-22s T %4d F %4d prec %.3f | U (pseudo pool) %6d' % (emb, k, T, F, T / max(1, T + F), U))
