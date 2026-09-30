"""Reviewer check (read-only): recompute r2_dup_an / r2_pair_census_or / r2_endstart summaries from their json dumps."""
import os, sys, json
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/official/src')
from collections import Counter, defaultdict
import numpy as np
from tracking_cellmot.metrics import summarise

RR = json.load(open('/workspace/cl/ideas/r2_dup_an.json'))
print('dup_an movies', len(RR), Counter(r['movie'][:4] for r in RR), Counter(r['set'] for r in RR))
rng = np.random.default_rng(1)
K_ = [rng.integers(0, len(RR), len(RR)) for _ in range(500)]
base_all = summarise([r['base'] for r in RR])
SK = lambda sel, k: sum(r[k] for r in sel)
print('P14 base all score %.5f edgeTP %d FP %d FN %d div %d/%d' % (base_all['score'], SK([r['base'] for r in RR],'edge_tp'), SK([r['base'] for r in RR],'edge_fp'), SK([r['base'] for r in RR],'edge_fn'), base_all['division_tp'], base_all['division_fp']))
for c in ['ee', 'ee_tt', 'fd', 'fdb', 'i2', 'zd', 'bub', 'fsib', 'tt2']:
    line = '%-6s n=%5d' % (c, sum(r['n'][c] for r in RR))
    for emb in ['44b6', '6bba', '']:
        sel = [r for r in RR if r['movie'].startswith(emb)]
        a = summarise([r['base'] for r in sel]); b = summarise([r['rows'][c] for r in sel])
        line += ' | %s %+.5f dTP%+d dFP%+d dN%+d dDiv%+d/%+d' % (emb or 'all', b['score'] - a['score'], SK([r['rows'][c] for r in sel],'edge_tp') - SK([r['base'] for r in sel],'edge_tp'), SK([r['rows'][c] for r in sel],'edge_fp') - SK([r['base'] for r in sel],'edge_fp'), SK([r['rows'][c] for r in sel],'num_pred_nodes') - SK([r['base'] for r in sel],'num_pred_nodes'), b['division_tp'] - a['division_tp'], b['division_fp'] - a['division_fp'])
    bs = [summarise([RR[j]['rows'][c] for j in k])['score'] - summarise([RR[j]['base'] for j in k])['score'] for k in K_]
    line += ' | CI [%+.5f,%+.5f]' % (np.quantile(bs, .025), np.quantile(bs, .975))
    cl = [r for r in RR if r['set'] in ('hold36', 'prev4')]
    line += ' | c40 %+.5f' % (summarise([r['rows'][c] for r in cl])['score'] - summarise([r['base'] for r in cl])['score'])
    lab = Counter()
    for r in RR: lab.update(r['lab'][c])
    print(line); print('      lab', dict(lab))
# node-count term check: N_pred / N_total
for emb in ['44b6', '6bba']:
    sel = [r for r in RR if r['movie'].startswith(emb)]
    np_ = sum(r['base']['num_pred_nodes'] for r in sel)
    ks = [k for k in sel[0]['base'].keys()]
    print(emb, 'keys', ks[:40])
    break

rows = json.load(open('/workspace/cl/ideas/r2_pair_census_or.json'))
print('census pairs', len(rows))
J = 0.93
def val(o): return o[0] - J * (o[0] + o[1] + o[2])
cls = Counter(); posn = Counter(); posv = defaultdict(float); divpos = Counter(); anydiv = Counter()
for r in rows:
    k = '-'.join(sorted([r['ta'], r['tb']])); cls[k] += 1
    va, vb = val(r['oa']), val(r['ob']); v = max(va, vb)
    if r['oa'][3] or r['oa'][4] or r['ob'][3] or r['ob'][4]: anydiv[k] += 1
    if v > 1e-9: posn[k] += 1; posv[k] += v
print('classes', cls.most_common())
print('pos single deletion', {k: (posn[k], round(posv[k], 1)) for k in posn})
print('pairs with any division effect', dict(anydiv))
# coarse class grouping as in idea
def coarse(k):
    a, b = k.split('-')
    if 'F' in a or 'F' in b or a.endswith('d') or b.endswith('d'): return 'fork-related'
    return k
cc = Counter(coarse(k) for k in cls.elements())
print('coarse', cc.most_common())
tot = sum(posv.values()); print('oracle total edge units %.1f ~ score %+.5f' % (tot, tot * 7.5e-6))
# direct pos with raw TP/FP/FN
dd = Counter()
for r in rows:
    va, vb = val(r['oa']), val(r['ob'])
    if max(va, vb) > 1e-9:
        o = r['oa'] if va >= vb else r['ob']
        for i, kk in enumerate(['tp', 'fp', 'fn', 'dtp', 'dfp']): dd[kk] += o[i]
print('oracle picks raw sums', dict(dd))

es = json.load(open('/workspace/cl/ideas/r2_endstart.json'))
print('endstart rows', len(es))
for cr in [1, 0]:
    for db in [(0, 3.5), (3.5, 5), (5, 7)]:
        sel = [r for r in es if r['created'] == cr and db[0] < r['d'] <= db[1]]
        print('created', cr, db, len(sel), dict(Counter(r['lab'] for r in sel)), {e: dict(Counter(r['lab'] for r in sel if r['m'].startswith(e))) for e in ['44b6', '6bba']})
