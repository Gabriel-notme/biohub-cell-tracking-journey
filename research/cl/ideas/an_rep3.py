import json
from collections import Counter
import numpy as np
R = json.load(open('/workspace/cl/ideas/an/p13_resid.json'))
print('GT nodes', len(R), 'matched', sum(r['matched'] for r in R))
for emb in ['44b6', '6bba']:
    X = [r for r in R if r['m'].startswith(emb)]
    M = [r for r in X if r['m_is_n']]; NM = [r for r in X if not r['matched'] and 7 <= r['d'] < 10 and not r['n_taken']]
    print(emb, 'n', len(X), 'matched', sum(r['matched'] for r in X), 'nearmiss7-10 free', len(NM), 'unmatched', sum(not r['matched'] for r in X))
    for nm, S in [('matched', M), ('nearmiss', NM)]:
        r = np.array([s['r'] for s in S]); rf = np.array([s['rf'] for s in S])
        print('  %-9s mean res (um) z %.2f y %.2f x %.2f | abs z %.2f y %.2f x %.2f | float mean z %.2f y %.2f x %.2f' % ((nm,) + tuple(r.mean(0)) + tuple(np.abs(r).mean(0)) + tuple(rf.mean(0))))
        # z component share of distance in near-miss
    if NM:
        r = np.array([s['r'] for s in NM]); d = np.linalg.norm(r, axis=1)
        print('  nearmiss |dz| frac>5um %.2f, dominant axis z %.2f' % ((np.abs(r[:, 0]) > 5).mean(), (np.abs(r[:, 0]) > np.linalg.norm(r[:, 1:], axis=1)).mean()))
        sp = [s['spike'] for s in NM if s['spike'] is not None]; gsp = [s['gspike'] for s in NM if s['gspike'] is not None]
        print('  nearmiss pred spike median %.2f p75 %.2f ; gt spike median %.2f' % (np.median(sp), np.quantile(sp, .75), np.median(gsp)))
        sp = [s['spike'] for s in M if s['spike'] is not None]; gsp = [s['gspike'] for s in M if s['gspike'] is not None]
        print('  matched  pred spike median %.2f p75 %.2f ; gt spike median %.2f' % (np.median(sp), np.quantile(sp, .75), np.median(gsp)))
        # velocity projection: residual along pred velocity direction
        for nm, S in [('matched', M), ('nearmiss', NM)]:
            proj = []
            for s in S:
                if s['pv'] is None: continue
                v = np.array(s['pv']); vn = np.linalg.norm(v)
                if vn < 1.5: continue
                proj.append(np.dot(np.array(s['r']), v / vn))
            print('  %-9s residual along pred velocity (|v|>=1.5um/f): n %d mean %.2f' % (nm, len(proj), np.mean(proj)))
        # gz distribution
        print('  nearmiss gz quantiles', np.quantile([s['gz'] for s in NM], [.1, .5, .9]).round(1), 'matched', np.quantile([s['gz'] for s in M], [.1, .5, .9]).round(1))
        print('  nearmiss n structure', Counter((s['npar'], s['nout']) for s in NM).most_common(6))
        print('  nearmiss t quantiles', np.quantile([s['t'] for s in NM], [.1, .5, .9]))
