"""Verifier: concentration of per-movie deltas for r3_tbx4 (reads rule_eval rows only, no GT)."""
import json, sys
import numpy as np
sys.path.insert(0, '/workspace/official/src')
import warnings; warnings.filterwarnings('ignore')
from tracking_cellmot.metrics import summarise
R = json.load(open('/workspace/cl/ideas/r3_tbx4_verify_rows.json'))
V = ['p0.5', 'p0.5+join', 'p0.8', 'p0.8+join']
base = {r['movie']: r for r in R if r['vi'] == 0}
ms = sorted(base)


def d(cur, mm):
    return summarise([cur[m] for m in mm])['score'] - summarise([base[m] for m in mm])['score']


def dedge(cur, mm):
    return summarise([cur[m] for m in mm])['adj_edge_jaccard'] - summarise([base[m] for m in mm])['adj_edge_jaccard']


for i, vn in enumerate(V, 1):
    cur = {r['movie']: r for r in R if r['vi'] == i}
    print('=== %s  all %+.5f  edge-only %+.5f' % (vn, d(cur, ms), dedge(cur, ms)))
    for emb in ['44b6', '6bba']:
        mm = [m for m in ms if m.startswith(emb)]
        W = sum(base[m]['edge_tp'] + base[m]['edge_fp'] + base[m]['edge_fn'] for m in mm)
        # per-movie exact leave-one-out influence on the embryo delta
        full = d(cur, mm)
        infl = {m: full - d(cur, [x for x in mm if x != m]) for m in mm}
        # approximate additive contribution on adj edge J
        c = {}
        for m in mm:
            w0 = base[m]['edge_tp'] + base[m]['edge_fp'] + base[m]['edge_fn']; w1 = cur[m]['edge_tp'] + cur[m]['edge_fp'] + cur[m]['edge_fn']
            c[m] = (w1 * cur[m]['adj_edge_jaccard'] - w0 * base[m]['adj_edge_jaccard']) / W
        cs = np.array(sorted(c.values(), reverse=True)); pos = cs[cs > 0].sum()
        top = sorted(mm, key=lambda m: -c[m])
        up = sum(1 for m in mm if cur[m]['adj_edge_jaccard'] > base[m]['adj_edge_jaccard'] + 1e-12)
        dn = sum(1 for m in mm if cur[m]['adj_edge_jaccard'] < base[m]['adj_edge_jaccard'] - 1e-12)
        dtp = sum(cur[m]['edge_tp'] - base[m]['edge_tp'] for m in mm); dfp = sum(cur[m]['edge_fp'] - base[m]['edge_fp'] for m in mm)
        dfn = sum(cur[m]['edge_fn'] - base[m]['edge_fn'] for m in mm)
        print('  %s n=%d delta %+.5f  up %d down %d  dTP %+d dFP %+d dFN %+d  sum_c %+.5f  top1/3/5/10 share of net %.2f/%.2f/%.2f/%.2f' % (
            emb, len(mm), full, up, dn, dtp, dfp, dfn, cs.sum(), cs[:1].sum() / cs.sum(), cs[:3].sum() / cs.sum(), cs[:5].sum() / cs.sum(), cs[:10].sum() / cs.sum()))
        for k in [1, 3, 5, 10]:
            print('    drop top-%d movies -> %s delta %+.5f' % (k, emb, d(cur, [m for m in mm if m not in top[:k]])))
        for m in top[:5]:
            print('    top %s set %s c %+.5f infl %+.5f dTP %+d dFP %+d dnodes %+d base_w %d' % (
                m, base[m]['set'], c[m], infl[m], cur[m]['edge_tp'] - base[m]['edge_tp'], cur[m]['edge_fp'] - base[m]['edge_fp'],
                cur[m]['num_pred_nodes'] - base[m]['num_pred_nodes'], base[m]['edge_tp'] + base[m]['edge_fp'] + base[m]['edge_fn']))
        worst = sorted(mm, key=lambda m: c[m])[:3]
        for m in worst:
            print('    worst %s set %s c %+.5f dTP %+d dFP %+d dnodes %+d' % (
                m, base[m]['set'], c[m], cur[m]['edge_tp'] - base[m]['edge_tp'], cur[m]['edge_fp'] - base[m]['edge_fp'], cur[m]['num_pred_nodes'] - base[m]['num_pred_nodes']))
    # all-199 drop top-k
    W = sum(base[m]['edge_tp'] + base[m]['edge_fp'] + base[m]['edge_fn'] for m in ms)
    ca = {m: ((cur[m]['edge_tp'] + cur[m]['edge_fp'] + cur[m]['edge_fn']) * cur[m]['adj_edge_jaccard'] - (base[m]['edge_tp'] + base[m]['edge_fp'] + base[m]['edge_fn']) * base[m]['adj_edge_jaccard']) / W for m in ms}
    top = sorted(ms, key=lambda m: -ca[m])
    print('  all: drop top-5 %+.5f  drop top-10 %+.5f  drop top-20 %+.5f' % (d(cur, [m for m in ms if m not in top[:5]]), d(cur, [m for m in ms if m not in top[:10]]), d(cur, [m for m in ms if m not in top[:20]])))
    # movies with any added-edge TP gain
    gain = [m for m in ms if cur[m]['edge_tp'] > base[m]['edge_tp']]
    print('  movies with dTP>0: %d / %d ; dTP distribution top10 %s' % (len(gain), len(ms), sorted([cur[m]['edge_tp'] - base[m]['edge_tp'] for m in ms], reverse=True)[:10]))
    # set-level
    print('  per set: ' + ' '.join('%s %+.5f' % (s, d(cur, [m for m in ms if base[m]['set'] == s])) for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b']))
    print('  leave-one-set-out: ' + ' '.join('%s %+.5f' % (s, d(cur, [m for m in ms if base[m]['set'] != s])) for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b']))
    # bootstrap per embryo (2000)
    rng = np.random.default_rng(1)
    for emb in ['44b6', '6bba']:
        mm = [m for m in ms if m.startswith(emb)]
        bs = []
        for _ in range(2000):
            k = rng.integers(0, len(mm), len(mm)); sm = [mm[j] for j in k]
            bs.append(summarise([cur[m] for m in sm])['score'] - summarise([base[m] for m in sm])['score'])
        bs = np.array(bs)
        print('  boot %s [%+.5f, %+.5f] P>0 %.3f' % (emb, np.quantile(bs, .025), np.quantile(bs, .975), (bs > 0).mean()))
