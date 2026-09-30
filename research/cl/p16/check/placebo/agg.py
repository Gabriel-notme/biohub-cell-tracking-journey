"""Aggregate placebo rows (key: placebo)."""
import sys, json, glob, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/official/src')
import numpy as np
from tracking_cellmot.metrics import summarise

RD = sys.argv[1] if len(sys.argv) > 1 else '/workspace/cl/p16/check/placebo/rows'
F = sorted(glob.glob(RD + '/*.json'))
INFO = {}; ROWS = {}
for f in F:
    d = json.load(open(f)); m = d['info']['movie']; INFO[m] = d['info']
    ROWS[m] = {r['var']: r for r in d['rows']}
ms = sorted(ROWS); print('movies', len(ms))
K = max(int(t[1:]) for t in ROWS[ms[0]] if t[0] == 'A' and t[1:].isdigit()) + 1
EK = ['edge_tp', 'edge_fp', 'edge_fn']; DK = ['division_tp', 'division_fp', 'division_fn']; CK = EK + DK + ['num_pred_nodes']

# ---- sanity vs stored rows
st = json.load(open('/workspace/cl/p16/rows_p17_full.json'))
sb = {r['movie']: r for r in st if r['vi'] == 0}; sc = {r['movie']: r for r in st if r['vi'] == 1}
bad15 = [m for m in ms if any(ROWS[m]['p15'][k] != sb[m][k] for k in CK)]
bad17 = [m for m in ms if any(ROWS[m]['p17file'][k] != sc[m][k] for k in CK)]
print('sanity: p15 rows != stored vi0:', len(bad15), bad15[:5], '| p17file != stored vi1:', len(bad17), bad17[:5])
print('sanity: P17 == deployed rules(P15):', sum(INFO[m]['p17_eq_rules'] for m in ms), '/', len(ms), '| P17 subset P15:', sum(INFO[m]['p17_subset_p15'] for m in ms))
if len(ms) < 199: print('WARNING: only', len(ms), 'movies')

base = {m: ROWS[m]['p15'] for m in ms}
E44 = [m for m in ms if m.startswith('44b6')]; E6 = [m for m in ms if m.startswith('6bba')]
C40 = [m for m in ms if INFO[m]['set'] in ('hold36', 'prev4')]
GROUPS = [('all', ms), ('44b6', E44), ('6bba', E6), ('clean40', C40)]


def sc_(rows, sel): return summarise([rows[m] for m in sel])['score']


def delta(rows, sel): return sc_(rows, sel) - sc_(base, sel)


def var(tag): return {m: ROWS[m][tag] for m in ms}


def mk(b, e_src, d_src, n_src):
    r = dict(b)
    for k in EK: r[k] = e_src[k]
    for k in DK: r[k] = d_src[k]
    r['num_pred_nodes'] = n_src['num_pred_nodes']
    nt = r['n_total']; ratio = (r['num_pred_nodes'] - nt) / nt
    den = r['edge_tp'] + r['edge_fp'] + r['edge_fn']; J = r['edge_tp'] / den if den else float('nan')
    r['edge_jaccard'] = J; r['total_node_ratio'] = ratio; r['adj_edge_jaccard'] = max(0., J * (1 - 0.1 * ratio)) if J == J else float('nan')
    return r


def parts(rows):
    """node-only / edge-only / div-only deltas (all movies)"""
    return {'node': delta({m: mk(base[m], base[m], base[m], rows[m]) for m in ms}, ms),
            'edge': delta({m: mk(base[m], rows[m], base[m], base[m]) for m in ms}, ms),
            'div': delta({m: mk(base[m], base[m], rows[m], base[m]) for m in ms}, ms)}


def cnt(rows): return {k: sum(rows[m][k] - base[m][k] for m in ms) for k in CK}


P17 = var('p17file')
obs = {g: delta(P17, sel) for g, sel in GROUPS}; obsp = parts(P17)
print('\n== P17 observed: ' + ' '.join('%s %+.5f' % (g, v) for g, v in obs.items()), '| parts', {k: round(v, 6) for k, v in obsp.items()}, cnt(P17))

print('\n== sequential rule increments (official metric, all 199):')
seq = [('cutdup', var('r_cd'), base), ('forkfrag', var('r_cdff'), var('r_cd')), ('shortbranch', P17, var('r_cdff'))]
for nm, cur, prev in seq:
    line = '%-12s' % nm
    for g, sel in GROUPS: line += ' %s %+.5f' % (g, sc_(cur, sel) - sc_(prev, sel))
    line += ' | counts ' + str({k: sum(cur[m][k] - prev[m][k] for m in ms) for k in CK})
    print(line)
for nm, tag in [('ffonly', 'r_ffonly'), ('sbonly', 'r_sbonly')]:
    cur = var(tag); print('%-12s' % nm, ' '.join('%s %+.5f' % (g, delta(cur, sel)) for g, sel in GROUPS), parts(cur), cnt(cur))


def report(prefix, obsd, label):
    D = {g: np.array([delta(var('%s%d' % (prefix, k)), sel) for k in range(K)]) for g, sel in GROUPS}
    print('\n== placebo %s (%s), K=%d' % (prefix, label, K))
    for g, _ in GROUPS:
        x = D[g]; o = obsd[g]; ge = int((x >= o - 1e-12).sum())
        print('  %-8s obs %+.5f | placebo mean %+.5f sd %.5f min %+.5f max %+.5f | #placebo>=obs %d/%d  one-sided p=%.3f | obs-mean %+.5f' % (
            g, o, x.mean(), x.std(ddof=1), x.min(), x.max(), ge, K, (1 + ge) / (K + 1), o - x.mean()))
    P = [parts(var('%s%d' % (prefix, k))) for k in range(K)]
    for p in ['node', 'edge', 'div']:
        x = np.array([q[p] for q in P]); print('  part %-4s placebo mean %+.6f sd %.6f min %+.6f max %+.6f' % (p, x.mean(), x.std(ddof=1), x.min(), x.max()))
    C = [cnt(var('%s%d' % (prefix, k))) for k in range(K)]
    print('  counts mean', {k: round(float(np.mean([c[k] for c in C])), 2) for k in CK})
    print('  counts per k:', [(c['edge_tp'], c['edge_fp'], c['division_tp'], c['division_fp'], c['num_pred_nodes']) for c in C])
    return D


DA = report('A', obs, 'cd->random unselected gap chains, ff->random unselected small fork-free comps, sb->random linear track-end tails')
DB = report('B', obs, 'same cd/ff placebo, sb->cut first L nodes of a daughter branch at a random unselected fork')
# component placebos
comp_obs = {'Acd': {g: delta(var('r_cd'), sel) for g, sel in GROUPS}, 'Aff': {g: delta(var('r_ffonly'), sel) for g, sel in GROUPS},
            'Asb': {g: delta(var('r_sbonly'), sel) for g, sel in GROUPS}, 'Bsb': {g: delta(var('r_sbonly'), sel) for g, sel in GROUPS}}
for pre in ['Acd', 'Aff', 'Asb', 'Bsb']: report(pre, comp_obs[pre], 'component-only vs the corresponding single rule on P15')

# ---- placebo matching quality
for fam in ['A', 'B']:
    need = sum(INFO[m]['rm']['all'] for m in ms)
    got = [sum(INFO[m]['placebo'][k][fam]['cd_nodes'] + INFO[m]['placebo'][k][fam]['ff_nodes'] + INFO[m]['placebo'][k][fam]['sb_nodes'] for m in ms) for k in range(K)]
    sh = {c: [sum(INFO[m]['placebo'][k][fam][c + '_short'] for m in ms) for k in range(K)] for c in ['cd', 'ff', 'sb']}
    print('\nmatching %s: P17 removed %d nodes; placebo removed min %d max %d; shortfall per comp (max over k) %s' % (fam, need, min(got), max(got), {c: max(v) for c, v in sh.items()}))
print('rule node totals', {c: sum(INFO[m]['rm'][c] for m in ms) for c in ['cd', 'ff', 'sb', 'all']})
print('movies with any placebo shortfall (A):', sorted({m for m in ms for k in range(K) for c in ['cd', 'ff', 'sb'] if INFO[m]['placebo'][k]['A'][c + '_short'] > 0})[:20])

# ---- division part: fork-level analysis
nf = sum(INFO[m]['forks15'] for m in ms); ntp = sum(len(INFO[m]['tpf']) for m in ms); nfp = sum(len(INFO[m]['fpf']) for m in ms)
print('\n== forks: P15 total forks %d, TP forks %d, FP forks %d (official score_divisions)' % (nf, ntp, nfp))
for c in ['cd', 'ff', 'sb', 'all']:
    L = [(m, p) for m in ms for p in INFO[m]['lost_forks'][c]]
    t = sum(1 for m, p in L if p in set(INFO[m]['tpf'])); f_ = sum(1 for m, p in L if p in set(INFO[m]['fpf']))
    print('  P17 %-3s lost forks %d: TP %d FP %d | %s' % (c, len(L), t, f_, [(m, p) for m, p in L if p in set(INFO[m]['fpf']) | set(INFO[m]['tpf'])]))
for fam in ['A', 'B']:
    res = []
    for k in range(K):
        L = [(m, p) for m in ms for p in INFO[m]['placebo'][k][fam + '_lost_forks']]
        res.append((len(L), sum(1 for m, p in L if p in set(INFO[m]['tpf'])), sum(1 for m, p in L if p in set(INFO[m]['fpf']))))
    print('  placebo %s lost forks (n, TP, FP) per k:' % fam, res)
# per-movie matched random-fork permutation (closed form, cheap, many draws): P17 loses n_m forks in movie m; draw n_m forks at random from
# that movie's P15 forks; approximate division change: TP fork lost -> tp-1, fn+1; FP fork lost -> fp-1
tp0 = sum(base[m]['division_tp'] for m in ms); fp0 = sum(base[m]['division_fp'] for m in ms); fn0 = sum(base[m]['division_fn'] for m in ms)
J = lambda tp, fp, fn: tp / (tp + fp + fn)
rng = np.random.default_rng(0); NP = 5000
for scope, cset in [('all P17 lost forks', 'all'), ('shortbranch lost forks', 'sb')]:
    need = {m: len(INFO[m]['lost_forks'][cset]) for m in ms}
    L = [(m, p) for m in ms for p in INFO[m]['lost_forks'][cset]]
    ot = sum(1 for m, p in L if p in set(INFO[m]['tpf'])); of = sum(1 for m, p in L if p in set(INFO[m]['fpf']))
    od = 0.1 * (J(tp0 - ot, fp0 - of, fn0 + ot) - J(tp0, fp0, fn0))
    sims = []; nfpv = []; ntpv = []
    # sample fork statuses from per-movie counts (forks15, |tpf|, |fpf|)
    for _ in range(NP):
        t = f_ = 0
        for m in ms:
            n = need[m]
            if n == 0: continue
            N = INFO[m]['forks15']; a = len(INFO[m]['tpf']); b = len(INFO[m]['fpf'])
            lab = rng.choice(N, size=min(n, N), replace=False)
            t += int((lab < a).sum()); f_ += int(((lab >= a) & (lab < a + b)).sum())
        sims.append(0.1 * (J(tp0 - t, fp0 - f_, fn0 + t) - J(tp0, fp0, fn0))); nfpv.append(f_); ntpv.append(t)
    sims = np.array(sims)
    print('  random-fork permutation (%s, per-movie matched n=%d): observed TP %d FP %d -> div part %+.5f | random mean %+.5f, E[TP lost] %.2f E[FP lost] %.2f, P(div part >= obs) = %.4f, P(TP lost = 0) = %.3f, P(FP lost >= %d) = %.3f' % (
        scope, sum(need.values()), ot, of, od, sims.mean(), np.mean(ntpv), np.mean(nfpv), (sims >= od - 1e-12).mean(), (np.array(ntpv) == 0).mean(), of, (np.array(nfpv) >= of).mean()))
json.dump({'obs': obs, 'A': {g: v.tolist() for g, v in DA.items()}, 'B': {g: v.tolist() for g, v in DB.items()}}, open(RD + '/../agg_out.json', 'w'))

