"""Per movie x rule evaluable changes across graph versions: which movies lose, and does the same movie lose on P15.
usage: permov.py src1,src2,...  """
import sys, json
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
import warnings; warnings.filterwarnings('ignore')
D = '/workspace/cl/p16/check2/xgraph/'
srcs = sys.argv[1].split(',')
O = ['base', 'cd', 'ff', 'st', 'tt', 'par', 'border']
KE = ['edge_tp', 'edge_fp', 'edge_fn']; KD = ['division_tp', 'division_fp', 'division_fn']
by = {}; det = {}
for s in srcs:
    by[s] = {}
    for r in json.load(open(D + 'rows_%s.json' % s)): by[s].setdefault(r['stage'], {})[r['movie']] = r
    det[s] = {d['movie']: d for d in json.load(open(D + 'det_%s.json' % s))}
ms = sorted(by[srcs[0]]['base'])
sb = {s: summarise(list(by[s]['base'].values()))['score'] for s in srcs}
def marg(s, a, c, m):  # exact marginal of rule step a->c in movie m, on top of stage a everywhere
    A = by[s][a]; C = by[s][c]
    return summarise([C[x] if x == m else A[x] for x in ms])['score'] - summarise(list(A.values()))['score']
for i in range(1, len(O)):
    a, c = O[i - 1], O[i]
    print('==== rule', c)
    lines = []
    for m in ms:
        chg = {s: {k: by[s][c][m][k] - by[s][a][m][k] for k in KE + KD if by[s][c][m][k] != by[s][a][m][k]} for s in srcs}
        if not any(chg.values()): continue
        mg = {s: marg(s, a, c, m) if chg[s] else None for s in srcs}
        lines.append((m, chg, mg))
    for m, chg, mg in sorted(lines, key=lambda x: min(v for v in x[2].values() if v is not None)):
        st = by[srcs[0]]['base'][m]['set']
        print('  %-15s %-7s ' % (m, st) + ' | '.join('%s %s %s' % (s, ('%+.6f' % mg[s]) if mg[s] is not None else '    .    ',
              ' '.join('%s%+d' % (k.replace('edge_', 'e').replace('division_', 'd'), v) for k, v in chg[s].items())) for s in srcs))
