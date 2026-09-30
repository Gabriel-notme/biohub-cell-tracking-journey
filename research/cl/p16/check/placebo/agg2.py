"""Follow-up: (a) P17 non-division part vs placebo A; (b) per-movie recombination of placebo draws (placebo deletions are independent
per movie, so any per-movie combination of the K draws is a valid placebo realisation) -> finer-resolution one-sided p-values."""
import sys, json, glob, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/official/src')
import numpy as np
from tracking_cellmot.metrics import summarise

RD = '/workspace/cl/p16/check/placebo/rows'
INFO = {}; ROWS = {}
for f in sorted(glob.glob(RD + '/*.json')):
    d = json.load(open(f)); m = d['info']['movie']; INFO[m] = d['info']; ROWS[m] = {r['var']: r for r in d['rows']}
ms = sorted(ROWS); K = 20
EK = ['edge_tp', 'edge_fp', 'edge_fn']; DK = ['division_tp', 'division_fp', 'division_fn']
base = {m: ROWS[m]['p15'] for m in ms}
GROUPS = [('all', ms), ('44b6', [m for m in ms if m.startswith('44b6')]), ('6bba', [m for m in ms if m.startswith('6bba')]),
          ('clean40', [m for m in ms if INFO[m]['set'] in ('hold36', 'prev4')])]
SB = {g: summarise([base[m] for m in sel])['score'] for g, sel in GROUPS}


def mk(b, e_src, d_src, n_src):
    r = dict(b)
    for k in EK: r[k] = e_src[k]
    for k in DK: r[k] = d_src[k]
    r['num_pred_nodes'] = n_src['num_pred_nodes']
    nt = r['n_total']; ratio = (r['num_pred_nodes'] - nt) / nt
    den = r['edge_tp'] + r['edge_fp'] + r['edge_fn']; J = r['edge_tp'] / den if den else float('nan')
    r['edge_jaccard'] = J; r['total_node_ratio'] = ratio; r['adj_edge_jaccard'] = max(0., J * (1 - 0.1 * ratio)) if J == J else float('nan')
    return r


def dl(rows, g, sel): return summarise([rows[m] for m in sel])['score'] - SB[g]


P17 = {m: ROWS[m]['p17file'] for m in ms}
P17nd = {m: mk(base[m], P17[m], base[m], P17[m]) for m in ms}  # P17 edges+nodes, P15 divisions
tests = [('P17 full vs A', P17, 'A'), ('P17 non-division part (edges+nodes) vs A', P17nd, 'A'),
         ('cutdup alone vs Acd', {m: ROWS[m]['r_cd'] for m in ms}, 'Acd'), ('forkfrag alone vs Aff', {m: ROWS[m]['r_ffonly'] for m in ms}, 'Aff'),
         ('shortbranch alone vs Asb', {m: ROWS[m]['r_sbonly'] for m in ms}, 'Asb'),
         ('shortbranch alone, non-division part vs Asb', {m: mk(base[m], ROWS[m]['r_sbonly'], base[m], ROWS[m]['r_sbonly']) for m in ms}, 'Asb')]
rng = np.random.default_rng(1); R = 4000
for name, obsrows, pre in tests:
    print('\n== %s' % name)
    for g, sel in GROUPS:
        o = dl(obsrows, g, sel)
        ex = np.array([dl({m: ROWS[m]['%s%d' % (pre, k)] for m in sel}, g, sel) for k in range(K)])
        ge = int((ex >= o - 1e-12).sum())
        rec = []
        for _ in range(R):
            ks = rng.integers(0, K, len(sel))
            rec.append(summarise([ROWS[m]['%s%d' % (pre, k)] for m, k in zip(sel, ks)])['score'] - SB[g])
        rec = np.array(rec); gr = int((rec >= o - 1e-12).sum())
        print('  %-8s obs %+.5f | exact K=20: mean %+.5f sd %.5f max %+.5f, >=obs %d/20, p=%.3f | recombined R=%d: mean %+.5f sd %.5f q99 %+.5f max %+.5f, >=obs %d, p=%.5f | z=%.1f' % (
            g, o, ex.mean(), ex.std(ddof=1), ex.max(), ge, (1 + ge) / (K + 1), R, rec.mean(), rec.std(ddof=1), np.quantile(rec, .99), rec.max(), gr, (1 + gr) / (R + 1),
            (o - rec.mean()) / rec.std(ddof=1)))
# per-movie specificity of edges: FP/TP edges removed per 1000 deleted nodes
def rate(rowsets, nodes):
    tp = sum(rowsets[m]['edge_tp'] - base[m]['edge_tp'] for m in ms); fp = sum(rowsets[m]['edge_fp'] - base[m]['edge_fp'] for m in ms)
    return tp, fp
print('\nedge TP/FP change: P17', rate(P17, 0), '| placebo A per k', [rate({m: ROWS[m]['A%d' % k] for m in ms}, 0) for k in range(K)])
print('cutdup', rate({m: ROWS[m]['r_cd'] for m in ms}, 0), '| Acd per k', [rate({m: ROWS[m]['Acd%d' % k] for m in ms}, 0) for k in range(K)])
print('forkfrag', rate({m: ROWS[m]['r_ffonly'] for m in ms}, 0), '| Aff per k', [rate({m: ROWS[m]['Aff%d' % k] for m in ms}, 0) for k in range(K)])
print('shortbranch', rate({m: ROWS[m]['r_sbonly'] for m in ms}, 0), '| Asb per k', [rate({m: ROWS[m]['Asb%d' % k] for m in ms}, 0) for k in range(K)])
# forks: status of shortbranch-selected forks by embryo, base FP/TP rates
for e in ['44b6', '6bba']:
    sel = [m for m in ms if m.startswith(e)]
    nf = sum(INFO[m]['forks15'] for m in sel); tp = sum(len(INFO[m]['tpf']) for m in sel); fp = sum(len(INFO[m]['fpf']) for m in sel)
    L = [(m, p) for m in sel for p in INFO[m]['lost_forks']['sb']]
    print('%s: P15 forks %d (TP %d, FP %d, FP rate %.4f, TP rate %.4f) | shortbranch-removed forks %d: TP %d FP %d (FP rate %.4f)' % (
        e, nf, tp, fp, fp / nf, tp / nf, len(L), sum(p in set(INFO[m]['tpf']) for m, p in L), sum(p in set(INFO[m]['fpf']) for m, p in L),
        sum(p in set(INFO[m]['fpf']) for m, p in L) / max(1, len(L))))
from scipy.stats import fisher_exact, binomtest
nf = sum(INFO[m]['forks15'] for m in ms); tp = sum(len(INFO[m]['tpf']) for m in ms); fp = sum(len(INFO[m]['fpf']) for m in ms)
n = 219
print('FP enrichment among shortbranch forks: 2/%d vs rest %d/%d: Fisher one-sided p=%.3f; TP depletion 0/%d vs rest %d/%d: Fisher one-sided p=%.4f' % (
    n, fp - 2, nf - n, fisher_exact([[2, n - 2], [fp - 2, nf - n - fp + 2]], alternative='greater')[1], n, tp, nf - n, fisher_exact([[0, n], [tp, nf - n - tp]], alternative='less')[1]))
print('Poisson view: expected FP-fork removals per 199 movies at the observed rate 2/219 x 219 = 2; P(0 events | mean 2) = %.3f' % np.exp(-2))

