"""Exact official-metric check of the stress test (44b6-trained junk prune applied to 6bba P8 graphs):
re-score pruned graphs with the official evaluate() and compare to the table approximation used by jp_xemb/jp_cap/jp_rel."""
import os, sys, pickle, json
os.environ.setdefault('POLARS_MAX_THREADS', '1'); os.environ['OMP_NUM_THREADS'] = '16'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/cl')
import numpy as np
from multiprocessing import Pool
L = lambda t: pickle.load(open('/workspace/cl/jp/%s.pkl' % t, 'rb'))
SETS = ['p8_t127a', 'p8_t127b', 'p8_audit32', 'p8_hold36', 'p8_prev4']
ALL = []
for t in SETS:
    for r in L(t): r['set'] = t; ALL.append(r)
GDIR = {'p8_hold36': '/workspace/cl/ps_p8_hold36/graphs', 'p8_prev4': '/workspace/cl/ps_p8_prev4/graphs', 'p8_audit32': '/workspace/cl/ps_p8_audit32/graphs'}


def job(a):
    movie, gdir, Xtab, sels = a
    import evalx, jprune
    nodes, edges = evalx.load_graph_json('%s/%s.json' % (gdir, movie))
    trk, X = jprune.features(nodes, edges)
    xdiff = float(np.nanmax(np.abs(X - Xtab))) if X.shape == Xtab.shape else -1.
    out = {'movie': movie, 'xdiff': xdiff, 'rows': {}}
    out['rows']['base'] = evalx.score_movie(movie, nodes, edges)
    for name, sel in sels.items():
        drop = set(n for c, s in zip(trk, sel) if s for n in c)
        nn = {n: v for n, v in nodes.items() if n not in drop}
        ee = [e for e in edges if int(e['source_id']) not in drop and int(e['target_id']) not in drop]
        row = evalx.score_movie(movie, nn, ee); row['removed'] = len(drop); out['rows'][name] = row
    return out


def score(rows):
    w = np.array([x['edge_tp'] + x['edge_fp'] + x['edge_fn'] for x in rows]); a = np.array([x['adj_edge_jaccard'] for x in rows])
    dt = sum(x['division_tp'] for x in rows); dd = dt + sum(x['division_fp'] + x['division_fn'] for x in rows)
    return (w * a).sum() / w.sum() + (0.1 * dt / dd if dd else 0)


def table(rs, name):
    n0 = d0 = n1 = d1 = 0.
    for r in rs:
        m = 1 - 0.1 * (r['npred'] - r['ntot']) / r['ntot']; w = r['TP'] + r['FP'] + r['FN']; sel = r['sels'][name]
        n = r['len'][sel].sum(); k = r['tp'][sel].sum(); f = r['fp'][sel].sum()
        n0 += r['TP'] * m; d0 += w; n1 += (r['TP'] - k) * (m + 0.1 * n / r['ntot']); d1 += w - f
    return n1 / d1 - n0 / d0


if __name__ == '__main__':
    import lightgbm as lgb
    params = dict(objective='poisson', learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=16)
    fit = lambda rs: lgb.train(params, lgb.Dataset(np.concatenate([r['X'] for r in rs]), np.concatenate([r['tp'] for r in rs])), 300)
    a44 = [r for r in ALL if r['movie'].startswith('44b6')]
    t44 = [r for r in a44 if r['set'] in ('p8_t127a', 'p8_t127b', 'p8_audit32')]
    m_all = fit(a44); m_tr = fit(t44)
    tp_ref = float(np.median([r['TP'] for r in t44])); ratio = float(np.median([r['ntot'] / r['npred'] for r in t44]))
    print('44b6 train n=%d tp_ref %.0f ratio %.4f' % (len(t44), tp_ref, ratio))
    tgt = [r for r in ALL if r['movie'].startswith('6bba') and r['set'] in GDIR]
    jobs = []
    for r in tgt:
        pa = m_all.predict(r['X']); pt = m_tr.predict(r['X'])
        rate = pa / r['len']; rbar = pa.sum() / r['len'].sum()
        nh = r['npred'] * ratio; mh = 1 - 0.1 * (r['npred'] - nh) / nh; be = 0.1 * tp_ref / nh / mh
        absel = pt < 1.0 * r['len'] * be
        cap = absel.copy()
        if r['len'][cap].sum() > 0.1 * r['npred']:
            idx = np.flatnonzero(cap); order = idx[np.argsort(pt[idx] / r['len'][idx])]
            keep = order[np.cumsum(r['len'][order]) <= 0.1 * r['npred']]; cap = np.zeros_like(cap); cap[keep] = True
        sels = {'rel05': rate < 0.05 * rbar, 'rel10': rate < 0.10 * rbar, 'abs_cap10': cap, 'abs': absel}
        r['sels'] = sels
        jobs.append((r['movie'], GDIR[r['set']], r['X'], sels))
    with Pool(16) as p: OUT = p.map(job, jobs)
    pickle.dump(OUT, open('exact_out.pkl', 'wb'))
    byname = {o['movie']: o for o in OUT}
    print('max feature diff table vs recomputed:', max(o['xdiff'] for o in OUT))
    for grp, rs in [('clean20', [r for r in tgt if r['set'] != 'p8_audit32']), ('audit16', [r for r in tgt if r['set'] == 'p8_audit32']), ('all36', tgt)]:
        base = score([byname[r['movie']]['rows']['base'] for r in rs])
        print('%-8s base %.5f | ' % (grp, base) + ' | '.join('%s official %+.5f table %+.5f' % (nm, score([byname[r['movie']]['rows'][nm] for r in rs]) - base, table(rs, nm)) for nm in ['rel05', 'rel10', 'abs_cap10', 'abs']))
    cache = {}
    for s in ['hold36', 'prev4', 'audit32']:
        for x in json.load(open('/workspace/cl/rev/p8_%s.json' % s)): cache[x['movie']] = x
    print('max |base adjJ - cached p8 adjJ|:', max(abs(o['rows']['base']['adj_edge_jaccard'] - cache[o['movie']]['adj_edge_jaccard']) for o in OUT))
