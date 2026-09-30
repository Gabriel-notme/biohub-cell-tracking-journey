import json, numpy as np
from collections import Counter
R = json.load(open('/workspace/cl/ideas/rv_ll_xfer_rows.json'))
L = [r for r in R if r['kind'] == 'long']; P = [r for r in R if r['kind'] == 'p13']
q = lambda v: np.round(np.quantile(np.array(v, float), [.1, .25, .5, .75, .9]), 2).tolist() if len(v) else []
print('long links', len(L), Counter((r['lab'], r['movie'][:4]) for r in L))
print('b5free', Counter((r['b5free'], r['lab']) for r in L))
for emb in ['44b6', '6bba']:
    for lab in ['TP', 'FP', 'U']:
        x = [r for r in L if r['movie'][:4] == emb and r['lab'] == lab]
        if not x: continue
        xr = [r for r in x if r['res'] is not None]
        print(emb, lab, len(x), 'fe', q([r['fe'] for r in x]), 'd', q([r['d'] for r in x]), '| res', len(xr), q([r['res'] for r in xr]),
              '| flowmag', q([r['fm'] for r in xr]), '| gmed', q([r['gmed'] for r in x if r['gmed'] is not None]))
# flow-consistency fraction: residual < 7um (matching radius)
for emb in ['44b6', '6bba']:
    for lab in ['TP', 'U']:
        xr = [r for r in L if r['movie'][:4] == emb and r['lab'] == lab and r['res'] is not None]
        if xr: print('flow-consistent(res<7)', emb, lab, round(np.mean([r['res'] < 7 for r in xr]), 3), 'res<10', round(np.mean([r['res'] < 10 for r in xr]), 3), len(xr))
# reference: existing P13 edges by length bin
print('--- existing P13 edges: fe and residual by length bin and label ---')
for emb in ['44b6', '6bba']:
    for lo, hi in [(0, 4), (4, 8), (8, 11), (11, 14), (14, 40)]:
        for lab in ['TP', 'FP', 'U']:
            x = [r for r in P if r['movie'][:4] == emb and r['lab'] == lab and lo <= r['d'] < hi]
            if len(x) < 5: continue
            xr = [r for r in x if r['res'] is not None]
            print(emb, '%2d-%2d' % (lo, hi), lab, len(x), 'fe', q([r['fe'] for r in x]), 'P(fe>=.5)', round(np.mean([r['fe'] >= .5 for r in x]), 3),
                  'res', q([r['res'] for r in xr]), 'res<7', round(np.mean([r['res'] < 7 for r in xr]), 3) if xr else None)
