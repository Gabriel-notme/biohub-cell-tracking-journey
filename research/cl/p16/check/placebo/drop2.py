import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/official/src')
import numpy as np
from tracking_cellmot.metrics import summarise
R = json.load(open('/workspace/cl/p16/rows_p17_full.json'))
base = {r['movie']: r for r in R if r['vi'] == 0}; cur = {r['movie']: r for r in R if r['vi'] == 1}
ms = sorted(base); drop = {'44b6_c50204e0', '6bba_5c824876'}
rng = np.random.default_rng(0)
def d(sel): return summarise([cur[m] for m in sel])['score'] - summarise([base[m] for m in sel])['score']
for lab, sel in [('all', ms), ('all minus 2 div movies', [m for m in ms if m not in drop])]:
    for emb in ['', '44b6', '6bba']:
        s2 = [m for m in sel if m.startswith(emb)]
        bs = np.array([d([s2[j] for j in rng.integers(0, len(s2), len(s2))]) for _ in range(1000)])
        c40 = [m for m in s2 if base[m]['set'] in ('hold36', 'prev4')]
        print('%-24s %-5s n=%3d delta %+.5f CI [%+.5f, %+.5f] P(<=0) %.3f | clean40 %+.5f' % (lab, emb or 'all', len(s2), d(s2), np.quantile(bs, .025), np.quantile(bs, .975), (bs <= 0).mean(), d(c40)))
# per-division-event value
b = summarise([base[m] for m in ms]); print('P15 div', b['division_tp'], b['division_fp'], b['division_fn'], 'divJ', b['division_jaccard'])
J = lambda t, f, n: t / (t + f + n)
t, f, n = b['division_tp'], b['division_fp'], b['division_fn']
print('value of one FP fork removal %+.5f, of one TP fork loss %+.5f' % (0.1 * (J(t, f - 1, n) - J(t, f, n)), 0.1 * (J(t - 1, f, n + 1) - J(t, f, n))))
