"""chk2 xgraph analysis of rows_<src>.json / det_<src>.json written by xg.py.
usage: ana.py <src> [drop]   (drop = 1 prints exact drop-top-k by per-movie marginal)"""
import sys, json
sys.path.insert(0, '/workspace/official/src')
import numpy as np
from tracking_cellmot.metrics import summarise
import warnings; warnings.filterwarnings('ignore')
D = '/workspace/cl/p16/check2/xgraph/'
src = sys.argv[1]; drop = len(sys.argv) > 2 and sys.argv[2] == '1'
R = json.load(open(D + 'rows_%s.json' % src)); DET = {d['movie']: d for d in json.load(open(D + 'det_%s.json' % src))}
ST = ['base', 'full', 'cd', 'ff', 'st', 'tt', 'par', 'border']
by = {st: {r['movie']: r for r in R if r['stage'] == st} for st in ST}
base, cur = by['base'], by['full']
ms = sorted(base)
KE = ['edge_tp', 'edge_fp', 'edge_fn']; KD = ['division_tp', 'division_fp', 'division_fn']; KA = KE + KD + ['num_pred_nodes']
Ssc = lambda rs: summarise(rs)['score']
def d(sel, B=base, C=cur): return Ssc([C[m] for m in sel]) - Ssc([B[m] for m in sel]) if sel else 0.0
E44 = [m for m in ms if m.startswith('44b6')]; E6 = [m for m in ms if m.startswith('6bba')]
C40 = [m for m in ms if base[m]['set'] in ('hold36', 'prev4')]; C72 = [m for m in ms if base[m]['set'] in ('hold36', 'prev4', 'audit32')]
rng = np.random.default_rng(0)
def boot(sel, B=base, C=cur, n=1000):
    o = np.array([d([sel[i] for i in rng.integers(0, len(sel), len(sel))], B, C) for _ in range(n)])
    return np.quantile(o, .025), np.quantile(o, .975), (o > 0).mean()
print('==== src', src, 'movies', len(ms), '| stepwise == module output for all movies:', all(DET[m]['same_as_module'] for m in ms))
sb = summarise([base[m] for m in ms]); sc = summarise([cur[m] for m in ms])
print('base score %.5f (adjE %.5f divJ %.4f div tp/fp/fn %d/%d/%d)' % (sb['score'], sb['adj_edge_jaccard'], sb['division_jaccard'], sb['division_tp'], sb['division_fp'], sb['division_fn']))
lo, hi, p = boot(ms)
print('FULL all %+.5f CI [%+.5f, %+.5f] P>0 %.3f | 44b6 %+.5f | 6bba %+.5f | clean40 %+.5f | B5clean72 %+.5f' % (d(ms), lo, hi, p, d(E44), d(E6), d(C40), d(C72)))
for lab, sel in [('44b6', E44), ('6bba', E6), ('clean40', C40)]:
    lo, hi, p = boot(sel, n=600); print('   %-8s %+.5f CI [%+.5f, %+.5f] P>0 %.3f' % (lab, d(sel), lo, hi, p))
print('   per set: ' + ' '.join('%s %+.5f' % (s, d([m for m in ms if base[m]['set'] == s])) for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b']))
print('   count deltas:', {k: sum(cur[m][k] - base[m][k] for m in ms) for k in KA})
F = {k: sum(DET[m]['full'][k] if isinstance(DET[m]['full'][k], int) else len(DET[m]['full'][k]) for m in ms) for k in ['tp_lost', 'tp_gained', 'div_lost', 'div_gained', 'fpf_removed', 'fpf_added']}
print('   GT-level: TP GT edges lost %d gained %d | GT divisions lost %d gained %d | FP forks removed %d added %d' % tuple(F[k] for k in ['tp_lost', 'tp_gained', 'div_lost', 'div_gained', 'fpf_removed', 'fpf_added']))
for m in ms:
    f = DET[m]['full']
    if f['div_lost'] or f['div_gained']: print('     DIV change', m, base[m]['set'], 'lost', f['div_lost'], 'gained', f['div_gained'])
# decomposition
def mk(b, e_src, d_src, n_src):
    r = dict(b)
    for k in KE: r[k] = e_src[k]
    for k in KD: r[k] = d_src[k]
    r['num_pred_nodes'] = n_src['num_pred_nodes']
    nt = r['n_total']; ratio = (r['num_pred_nodes'] - nt) / nt
    den = r['edge_tp'] + r['edge_fp'] + r['edge_fn']; J = r['edge_tp'] / den if den else float('nan')
    r['edge_jaccard'] = J; r['total_node_ratio'] = ratio; r['adj_edge_jaccard'] = max(0., J * (1 - 0.1 * ratio)) if J == J else float('nan')
    return r
err = max(abs(mk(cur[m], cur[m], cur[m], cur[m])['adj_edge_jaccard'] - cur[m]['adj_edge_jaccard']) for m in ms if cur[m]['adj_edge_jaccard'] == cur[m]['adj_edge_jaccard'])
V = {'node-count-only': {m: mk(base[m], base[m], base[m], cur[m]) for m in ms}, 'edge-counts-only': {m: mk(base[m], cur[m], base[m], base[m]) for m in ms},
     'division-only': {m: mk(base[m], base[m], cur[m], base[m]) for m in ms}}
print('   decomposition (recompute err %.1e):' % err)
tot = {}
for lab, sel in [('all', ms), ('44b6', E44), ('6bba', E6), ('clean40', C40)]:
    parts = {k: d(sel, base, v) for k, v in V.items()}; t = d(sel)
    print('     %-8s total %+.6f = node %+.6f + edge %+.6f + div %+.6f + interaction %+.6f' % (lab, t, parts['node-count-only'], parts['edge-counts-only'], parts['division-only'], t - sum(parts.values())))
    tot[lab] = parts
lo, hi, p = boot(ms, base, V['edge-counts-only']); print('     edge-counts-only all CI [%+.6f, %+.6f] P>0 %.3f' % (lo, hi, p))
for lab, sel in [('44b6', E44), ('6bba', E6), ('clean40', C40)]:
    lo, hi, p = boot(sel, base, V['edge-counts-only'], 600); print('     edge-counts-only %-8s %+.6f CI [%+.6f, %+.6f] P>0 %.3f' % (lab, d(sel, base, V['edge-counts-only']), lo, hi, p))
# per rule incremental
print('   per-rule incremental (stage vs previous stage; all / 44b6 / 6bba / clean40 | nodes rm, rm matched, edges rm tp/fp/nonvalid, TP GT edges lost/gained, divs lost/gained, FP forks rm/added):')
O = ['base', 'cd', 'ff', 'st', 'tt', 'par', 'border']
for i in range(1, len(O)):
    a, c = by[O[i - 1]], by[O[i]]; ru = O[i]
    agg = {k: sum(DET[m]['rules'][ru][k] if isinstance(DET[m]['rules'][ru][k], int) else len(DET[m]['rules'][ru][k]) for m in ms)
           for k in ['n_rm', 'n_rm_matched', 'rm_tp_edges', 'rm_fp_edges', 'rm_nonvalid_edges', 'tp_lost', 'tp_gained', 'div_lost', 'div_gained', 'fpf_removed', 'fpf_added']}
    nd = {m: mk(a[m], a[m], a[m], c[m]) for m in ms}
    print('     %-6s %+.6f / %+.6f / %+.6f / %+.6f (node-only %+.6f, edge-only %+.6f) | rm %d (matched %d) edges tp %d fp %d nv %d | TPGT -%d +%d | div -%d +%d | fpf -%d +%d | movies touched %d' % (
        ru, d(ms, a, c), d(E44, a, c), d(E6, a, c), d(C40, a, c), d(ms, a, nd), d(ms, a, {m: mk(a[m], c[m], a[m], a[m]) for m in ms}),
        agg['n_rm'], agg['n_rm_matched'], agg['rm_tp_edges'], agg['rm_fp_edges'], agg['rm_nonvalid_edges'], agg['tp_lost'], agg['tp_gained'], agg['div_lost'], agg['div_gained'],
        agg['fpf_removed'], agg['fpf_added'], sum(1 for m in ms if DET[m]['rules'][ru]['n_rm'])))
    for m in ms:
        rr = DET[m]['rules'][ru]
        if rr['div_lost'] or rr['div_gained'] or rr['tpf_removed_nodes']:
            print('        %s %s %s div lost %s gained %s, TP fork nodes removed %s, div counts %s' % (ru, m, base[m]['set'], rr['div_lost'], rr['div_gained'], rr['tpf_removed_nodes'],
                  {k: c[m][k] - a[m][k] for k in KD if c[m][k] != a[m][k]}))
# per movie
ch = [m for m in ms if any(cur[m][k] != base[m][k] for k in KE)]
up = sum(1 for m in ch if (cur[m]['edge_tp'] - base[m]['edge_tp']) - 0.93 * (cur[m]['edge_fp'] - base[m]['edge_fp']) > 0)
dn = sum(1 for m in ch if (cur[m]['edge_tp'] - base[m]['edge_tp']) - 0.93 * (cur[m]['edge_fp'] - base[m]['edge_fp']) < 0)
print('   movies with edge-count change %d: net-edge up %d down %d; movies with TP edge loss (net) %d, with TP GT lost (gross) %d' % (
    len(ch), up, dn, sum(1 for m in ms if cur[m]['edge_tp'] < base[m]['edge_tp']), sum(1 for m in ms if DET[m]['full']['tp_lost'])))
if drop:
    sbase = Ssc([base[m] for m in ms]); tot = d(ms)
    marg = {m: Ssc([cur[x] if x == m else base[x] for x in ms]) - sbase for m in ms}
    top = sorted(ms, key=lambda m: marg[m], reverse=True)
    print('   exact marginals: total %+.6f sum %+.6f | positive %d zero %d negative %d | top10 share %.1f%%' % (tot, sum(marg.values()), sum(v > 1e-12 for v in marg.values()),
          sum(abs(v) <= 1e-12 for v in marg.values()), sum(v < -1e-12 for v in marg.values()), 100 * sum(marg[m] for m in top[:10]) / tot))
    for m in top[:10]:
        print('     %-16s %-7s marg %+.6f (%.1f%%) %s' % (m, base[m]['set'], marg[m], 100 * marg[m] / tot, ' '.join('%s%+d' % (k, cur[m][k] - base[m][k]) for k in KA if cur[m][k] != base[m][k])))
    print('     bottom 5:', ' '.join('%s %+.6f' % (m, marg[m]) for m in top[-5:]))
    for k in [0, 2, 5, 10, 20]:
        sel = [m for m in ms if m not in set(top[:k])]; lo, hi, p = boot(sel)
        s44 = [m for m in sel if m.startswith('44b6')]; s6 = [m for m in sel if m.startswith('6bba')]; c40 = [m for m in sel if m in set(C40)]
        eo = V['edge-counts-only']
        print('     drop top%-2d: all %+.6f CI [%+.6f, %+.6f] P>0 %.3f | 44b6 %+.6f 6bba %+.6f clean40 %+.6f | edge-only %+.6f' % (k, d(sel), lo, hi, p, d(s44), d(s6), d(c40), d(sel, base, eo)))
    for e, sel0 in [('44b6', E44), ('6bba', E6)]:
        te = sorted(sel0, key=lambda m: marg[m], reverse=True)
        for k in [2, 5, 10]:
            s2 = [m for m in sel0 if m not in set(te[:k])]; lo, hi, p = boot(s2, n=600)
            print('     %s drop own top%-2d: %+.6f CI [%+.6f, %+.6f] P>0 %.3f' % (e, k, d(s2), lo, hi, p))
    # drop-top-k by marginal of the evaluable (edge-count) part only
    eo = V['edge-counts-only']; se = Ssc([base[m] for m in ms])
    me = {m: Ssc([eo[x] if x == m else base[x] for x in ms]) - se for m in ms}
    te = sorted(ms, key=lambda m: me[m], reverse=True)
    print('   edge-only exact marginals: total %+.6f, positive %d negative %d' % (d(ms, base, eo), sum(v > 1e-12 for v in me.values()), sum(v < -1e-12 for v in me.values())))
    for k in [0, 2, 5, 10, 20]:
        sel = [m for m in ms if m not in set(te[:k])]
        print('     edge-only drop top%-2d: edge-only %+.6f | full %+.6f' % (k, d(sel, base, eo), d(sel)))
