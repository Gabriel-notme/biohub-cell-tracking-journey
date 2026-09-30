"""Density stress: replay the exact P20 clean-up on the P15 graphs with all physical distances compressed by factor f
(module-level voxel scale S * f; equivalent to every radius / f, i.e. a denser embryo). Also a 'tiny FOV' border stress
(shape_yx = (0, 0): every node counts as border -> worst case of yx_border_stubs = all small no-fork components)."""
import sys
sys.path.insert(0, '/workspace/cl/p16/redteam_p20')
from rt_common import *
from multiprocessing import Pool
FS = [1.0, 0.875, 0.8, 0.7, 0.6, 0.5]
RULES = ['cutdup', 'forkfrag', 'start_trim', 'term_trim', 'par_dup', 'border']


def one(sm):
    s, m = sm
    n15, e15, _ = load(p15p(s, m)); shp = zshape(s, m); rp = refp(s, m)
    F15 = forks(e15); out = {'set': s, 'movie': m, 'n': len(n15), 'e': len(e15), 'forks': len(F15), 'res': {}}
    variants = [('f%.3f' % f, dict(fscale=f)) for f in FS] + [('tinyfov', dict(border_shape=(0, 0))), ('par4.0', dict(par_r=4.0)), ('par4.5', dict(par_r=4.5))]
    for key, kw in variants:
        tm = {}
        res = steps(n15, e15, rp, shp[-2:], timing=tm, **kw)
        prev = set(n15); by = {}; brk = {}
        for k, n, e in res:
            by[k] = len(prev - set(n)); prev = set(n)
            Fk = forks(e)
            for p, c in F15.items():
                if Fk.get(p) != c and p not in brk: brk[p] = k
        fe = res[-1][2]
        out['res'][key] = dict(by=by, nodes_del=len(n15) - len(res[-1][1]), edges_del=len(set(pairs(e15)) - set(pairs(fe))),
                               fork_break=Counter(brk.values()), valid=validity(res[-1][1], fe, shp[0]), t=round(sum(tm.values()), 3))
    return out


if __name__ == '__main__':
    ms = movies()
    with Pool(6) as pool: rows = pool.map(one, ms, chunksize=1)
    Path('/workspace/cl/p16/redteam_p20/rt_b.json').write_text(json.dumps(rows))
    keys = list(rows[0]['res'])
    for key in keys:
        fn = np.array([r['res'][key]['nodes_del'] / r['n'] for r in rows]); fe = np.array([r['res'][key]['edges_del'] / max(1, r['e']) for r in rows])
        by = {k: sum(r['res'][key]['by'][k] for r in rows) for k in RULES}
        fb = Counter()
        for r in rows: fb.update(r['res'][key]['fork_break'])
        mx = {k: round(100 * max(r['res'][key]['by'][k] / r['n'] for r in rows), 3) for k in RULES}
        i = int(fn.argmax())
        print('%-8s nodes del tot %.3f%% max %.3f%% (%s) p99 %.3f%% | edges max %.3f%% | >2%%: %d | forks broken %d of %d %s | invalid %d' % (
            key, 100 * sum(r['res'][key]['nodes_del'] for r in rows) / sum(r['n'] for r in rows), 100 * fn.max(), rows[i]['movie'], 100 * np.percentile(fn, 99), 100 * fe.max(),
            int((fn > 0.02).sum()), sum(fb.values()), sum(r['forks'] for r in rows), dict(fb), sum(1 for r in rows if r['res'][key]['valid'])))
        print('          by rule totals %s | per-movie max %% %s' % (by, mx))
