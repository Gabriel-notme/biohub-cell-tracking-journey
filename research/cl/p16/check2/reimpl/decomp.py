"""check2/reimpl: split a deletion rule's score delta into (a) the change of the unadjusted per-movie edge Jaccard and (b) the pure
node-count factor change of J_adj = J * f(N_pred). Counterfactual 'no node bonus': each movie keeps its BASE factor f0 = adj0/J0 and gets
adj = J1 * f0. Also: movie bootstrap CI of the no-bonus delta, and per-embryo / clean40 values.
usage: decomp.py <rows.json> [labels json list]"""
import sys, json
import numpy as np
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
import warnings; warnings.filterwarnings('ignore')

rows = json.load(open(sys.argv[1])); labels = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
base = {r['movie']: r for r in rows if r['vi'] == 0}; ms = sorted(base)
rng = np.random.default_rng(0); K = [rng.integers(0, len(ms), len(ms)) for _ in range(400)]


def nobonus(r1, r0):
    r = dict(r1)
    if r0['edge_jaccard'] > 0 and r0['adj_edge_jaccard'] == r0['adj_edge_jaccard']:
        r['adj_edge_jaccard'] = r1['edge_jaccard'] * (r0['adj_edge_jaccard'] / r0['edge_jaccard'])
    return r


for vi in sorted({r['vi'] for r in rows} - {0}):
    cur = {r['movie']: r for r in rows if r['vi'] == vi}
    nb = {m: nobonus(cur[m], base[m]) for m in ms}

    def D(sel, C):
        return summarise([C[m] for m in sel])['score'] - summarise([base[m] for m in sel])['score']
    full = D(ms, cur); cf = D(ms, nb)
    bs = [D([ms[j] for j in k], nb) for k in K]
    emb = {e: (D([m for m in ms if m.startswith(e)], cur), D([m for m in ms if m.startswith(e)], nb)) for e in ('44b6', '6bba')}
    cl = [m for m in ms if base[m]['set'] in ('hold36', 'prev4')]
    dn = sum(cur[m]['num_pred_nodes'] - base[m]['num_pred_nodes'] for m in ms)
    de = sum(cur[m]['edge_fp'] - base[m]['edge_fp'] for m in ms); dtp = sum(cur[m]['edge_tp'] - base[m]['edge_tp'] for m in ms)
    dfn = sum(cur[m]['edge_fn'] - base[m]['edge_fn'] for m in ms)
    lab = labels[vi - 1] if labels and vi - 1 < len(labels) else ''
    print('vi=%d %-60s total %+.5f | no-node-bonus %+.5f CI [%+.5f, %+.5f] | bonus share %+.5f | 44b6 %+.5f/%+.5f 6bba %+.5f/%+.5f | clean40 %+.5f/%+.5f | nodes %+d edgeTP %+d FP %+d FN %+d' % (
        vi, lab[:60], full, cf, np.quantile(bs, .025), np.quantile(bs, .975), full - cf, emb['44b6'][0], emb['44b6'][1], emb['6bba'][0], emb['6bba'][1],
        D(cl, cur), D(cl, nb), dn, dtp, de, dfn))

