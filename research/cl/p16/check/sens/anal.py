import sys, json
import numpy as np
sys.path.insert(0, '/workspace/official/src')
import warnings; warnings.filterwarnings('ignore')
from tracking_cellmot.metrics import summarise
tag = sys.argv[1]
R = json.load(open('/workspace/cl/p16/check/sens/rows_%s.json' % tag)); V = json.load(open('/workspace/cl/p16/check/sens/variants_%s.json' % tag))
by = {}
for r in R: by.setdefault(r['vi'], {})[r['movie']] = r
base = by[0]; ms = sorted(base)
rng = np.random.default_rng(0); K = [rng.integers(0, len(ms), len(ms)) for _ in range(300)]
def S(rows): return summarise(rows)['score']
def d(cur, ref, sel): return S([cur[m] for m in sel]) - S([ref[m] for m in sel])
def mix(cur, ref, part):  # counterfactual row: node term from one, edge counts from other
    out = {}
    for m in ms:
        a, b = cur[m], ref[m]
        if part == 'node':  # ref edges/divs, cur node count
            r = dict(b); r['num_pred_nodes'] = a['num_pred_nodes']; tnr = (a['num_pred_nodes'] - b['n_total']) / b['n_total']
            r['adj_edge_jaccard'] = max(0, b['edge_jaccard'] * (1 - 0.1 * tnr)); out[m] = r
    return out
E1 = [m for m in ms if m.startswith('44b6')]; E2 = [m for m in ms if m.startswith('6bba')]
C40 = [m for m in ms if base[m]['set'] in ('hold36', 'prev4')]
REST = [m for m in ms if m not in set(C40)]
print('%-22s %9s %9s %9s %9s %9s %21s %8s | %6s %5s %5s %5s %5s %6s | %9s %9s' % ('variant', 'all', '44b6', '6bba', 'clean40', 'rest159', 'CI', 'P>0', 'nodes', 'eTP', 'eFP', 'eFN', 'dTP', 'dFP', 'nodeOnly', 'vsP17'))
for vi in sorted(by):
    if vi == 0: continue
    cur = by[vi]
    bs = np.array([S([cur[ms[j]] for j in k]) - S([base[ms[j]] for j in k]) for k in K])
    sm = lambda key: sum(cur[m][key] - base[m][key] for m in ms)
    no = mix(cur, base, 'node')
    print('%-22s %+.5f %+.5f %+.5f %+.5f %+.5f [%+.5f,%+.5f] %.3f | %+6d %+5d %+5d %+5d %+5d %+6d | %+.5f %+.5f' % (
        json.dumps(V[vi]).replace(' ', ''), d(cur, base, ms), d(cur, base, E1), d(cur, base, E2), d(cur, base, C40), d(cur, base, REST),
        np.quantile(bs, .025), np.quantile(bs, .975), (bs > 0).mean(), sm('num_pred_nodes'), sm('edge_tp'), sm('edge_fp'), sm('edge_fn'), sm('division_tp'), sm('division_fp'),
        d(no, base, ms), d(cur, by[1], ms)))
