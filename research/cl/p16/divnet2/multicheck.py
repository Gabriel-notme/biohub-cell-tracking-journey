"""Stricter multiple-check review of a candidate vs a base on per-movie official-metric rows (strict.py format: vi=0 base, vi=1 cand).
usage: multicheck.py <rows.json> [family_size m=1] [label]
Gates (all must hold): G1 all-movie delta > 0 with paired movie-bootstrap (4000) CI lower bound > 0 at the Bonferroni level alpha=0.05/m
(two-sided); G2 each embryo delta > 0 with bootstrap P(>0) >= 0.95; G3 house bar strict.py S1-S7; G4 split-half replication: in >= 80%
of 2000 random embryo-stratified 50/50 movie splits BOTH halves have delta > 0; G5 sign test on movies whose evaluable counts changed:
more up than down and one-sided binomial p < 0.05/m. Info: CI lower bound at m = 1, 3, 5, 10, 20 (how many tried alternatives the
result would survive), per-set deltas, division TP/FP changes."""
import sys, json, subprocess
import numpy as np
from math import comb
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
import warnings; warnings.filterwarnings('ignore')
rows = json.load(open(sys.argv[1])); m = int(sys.argv[2]) if len(sys.argv) > 2 else 1; label = sys.argv[3] if len(sys.argv) > 3 else sys.argv[1]
B = {r['movie']: r for r in rows if r['vi'] == 0}; C = {r['movie']: r for r in rows if r['vi'] == 1}; ms = sorted(B)
rng = np.random.default_rng(2026)
d = lambda sel: summarise([C[x] for x in sel])['score'] - summarise([B[x] for x in sel])['score']
def boot(sel, n):
    sel = list(sel); return np.array([d([sel[i] for i in rng.integers(0, len(sel), len(sel))]) for _ in range(n)])
g = {}
dall = d(ms); b = boot(ms, 4000); a = 0.05 / m
lo = np.quantile(b, a / 2)
g['G1'] = dall > 0 and lo > 0
print('== MULTICHECK', label, 'family m=%d' % m)
print('  all n=%d d %+.5f  CI(alpha=%.4f) [%+.5f, %+.5f]  P>0 %.4f' % (len(ms), dall, a, lo, np.quantile(b, 1 - a / 2), (b > 0).mean()))
print('  CI lower bound by family size: ' + ' '.join('m=%d %+.5f' % (k, np.quantile(b, 0.025 / k)) for k in (1, 3, 5, 10, 20)))
ok = True
for e in sorted({x[:4] for x in ms}):
    sel = [x for x in ms if x.startswith(e)]; de = d(sel); pe = (boot(sel, 2000) > 0).mean(); ok &= de > 0 and pe >= .95
    print('  embryo %s n=%d d %+.5f P>0 %.3f' % (e, len(sel), de, pe))
g['G2'] = ok
for s in sorted({B[x]['set'] for x in ms}):
    sel = [x for x in ms if B[x]['set'] == s]
    print('  set %-8s n=%3d d %+.5f div TP %+d FP %+d' % (s, len(sel), d(sel), sum(C[x]['division_tp'] - B[x]['division_tp'] for x in sel), sum(C[x]['division_fp'] - B[x]['division_fp'] for x in sel)))
st = subprocess.run(['python3', '/workspace/cl/strict.py', sys.argv[1], '1', label], capture_output=True, text=True).stdout
g['G3'] = ' PASS ' in st.split('\n')[0] + ' '
print('  strict: ' + st.split('\n')[0])
both = 0; emb = {e: [x for x in ms if x.startswith(e)] for e in {x[:4] for x in ms}}
for _ in range(2000):
    h1, h2 = [], []
    for e, sel in emb.items():
        p = rng.permutation(len(sel)); k = len(sel) // 2; h1 += [sel[i] for i in p[:k]]; h2 += [sel[i] for i in p[k:]]
    both += d(h1) > 0 and d(h2) > 0
g['G4'] = both / 2000 >= .8
print('  split-half replication: both halves > 0 in %.3f of 2000 splits' % (both / 2000))
KEYS = ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn']
chg = [x for x in ms if any(C[x][k] != B[x][k] for k in KEYS)]
def contrib(x):
    w = B[x]['edge_tp'] + B[x]['edge_fp'] + B[x]['edge_fn']
    return (C[x]['adj_edge_jaccard'] - B[x]['adj_edge_jaccard']) * w + 0.5 * ((C[x]['division_tp'] - B[x]['division_tp']) - 0.4 * (C[x]['division_fp'] - B[x]['division_fp']))
up = sum(contrib(x) > 1e-9 for x in chg); dn = sum(contrib(x) < -1e-9 for x in chg); n = up + dn
p = sum(comb(n, k) for k in range(up, n + 1)) / 2 ** n if n else 1.0
g['G5'] = up > dn and p < 0.05 / m
print('  sign test on %d movies with evaluable changes: up %d down %d  one-sided p %.4f (gate < %.4f)' % (len(chg), up, dn, p, 0.05 / m))
print('RESULT', label, ' '.join('%s=%s' % kv for kv in g.items()), '->', 'PASS' if all(g.values()) else 'FAIL', flush=True)
