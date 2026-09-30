import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[_k] = '1'
sys.path.insert(0, '/workspace/official/src')
import warnings; warnings.filterwarnings('ignore')
from collections import Counter, defaultdict
from tracking_cellmot.metrics import summarise
D = [json.load(open(f)) for f in sorted(glob.glob('/workspace/cl/nm/rl_oracle/*.json'))]
print('movies', len(D))
groups = {'44b6': lambda d: d['movie'].startswith('44b6'), '6bba': lambda d: d['movie'].startswith('6bba'), 'all': lambda d: True,
          'clean40': lambda d: d['set'] in ('hold36', 'prev4'), 'c40_44b6': lambda d: d['set'] in ('hold36', 'prev4') and d['movie'].startswith('44b6'),
          'c40_6bba': lambda d: d['set'] in ('hold36', 'prev4') and d['movie'].startswith('6bba')}
V = list(D[0]['scores'])
print('%-16s' % 'variant' + ''.join('%22s' % g for g in groups))
for v in V:
    line = '%-16s' % v
    for g, f in groups.items():
        sel = [d for d in D if f(d)]
        b = summarise([d['scores']['base'] for d in sel]); c = summarise([d['scores'][v] for d in sel])
        dtp = sum(d['scores'][v]['edge_tp'] - d['scores']['base']['edge_tp'] for d in sel); dfp = sum(d['scores'][v]['edge_fp'] - d['scores']['base']['edge_fp'] for d in sel)
        line += '  %+.5f(%+d/%+d)' % (c['score'] - b['score'], dtp, dfp)
    print(line)
for g in ['44b6', '6bba', 'clean40']:
    sel = [d for d in D if groups[g](d)]
    cnt = Counter(); [cnt.update(d['cnt']) for d in sel]
    print('\n==', g, 'movies', len(sel), 'edges', dict(cnt))
    fc = Counter(); [fc.update(d['fncat']) for d in sel]
    print(' fncat', sorted(fc.items()))
    fx = Counter()
    for d in sel:
        for x in d['fixes']:
            fx[(x['typ'], 'pre' if x['fe'] >= 0 else 'nopre', 'flip' if x['flip'] else 'real', 'rmTP' if 'TP' in x['rml'] else ('rmFP%d' % x['rml'].count('FP')))] += 1
    for k, v in sorted(fx.items()): print('   ', k, v)
    pl = Counter(); [pl.update(d['provlab']) for d in sel]
    print(' provlab', sorted((k, v) for k, v in pl.items() if not k.startswith('b5|NE')))
    po = Counter(); [po.update(d['pool']) for d in sel]
    print(' pool', sorted(po.items()))
    mv = Counter()
    for d in sel:
        n = sum(1 for x in d['fixes'] if not x['typ'].startswith('div_') and not x['flip'] and 'TP' not in x['rml'])
        mv[min(n, 10)] += 1
    print(' movies by #real link fixes', sorted(mv.items()))
