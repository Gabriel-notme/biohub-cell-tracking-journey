import json
from collections import Counter, defaultdict
import numpy as np
R = json.load(open('/workspace/cl/ideas/r3_ma_tb_rows.json'))


def fb(t):
    if t <= 3: return 'f%d' % t
    if t >= 96: return 'f%d' % t
    return 'mid'


ORD = ['f0', 'f1', 'f2', 'f3', 'mid', 'f96', 'f97', 'f98', 'f99']
for emb in ['44b6', '6bba', '']:
    rr = [x for r in R if r['emb'].startswith(emb) for x in r['rows']]
    print('\n===== emb', emb or 'all')
    print('frame   n     match  md_mean  dz_mean dy_mean dx_mean |dz| |dy| |dx|   raw_d_mean(matched)  start/end frac')
    for k in ORD:
        m = [x for x in rr if fb(x[0]) == k and x[1] == 1]; u = [x for x in rr if fb(x[0]) == k and x[1] == 0]
        n = len(m) + len(u)
        if not n: continue
        a = np.array([x[2:6] for x in m]); rw = np.array([x[6] for x in m if x[6] >= 0])
        se = np.array([x[7:9] for x in m])
        print('%-5s %6d  %.4f  %.2f   %+.2f %+.2f %+.2f  %.2f %.2f %.2f   %.2f   %.3f/%.3f' % (k, n, len(m) / n, a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean(),
              np.abs(a[:, 1]).mean(), np.abs(a[:, 2]).mean(), np.abs(a[:, 3]).mean(), rw.mean() if len(rw) else -1, se[:, 0].mean(), se[:, 1].mean()))
    print('--- unmatched anatomy')
    for k in ORD:
        u = [x for x in rr if fb(x[0]) == k and x[1] == 0]
        if not u: continue
        n_all = len([x for x in rr if fb(x[0]) == k])
        how = Counter(x[4] for x in u)
        own = [x for x in u if x[4] == 'own']
        od = np.array([x[5] for x in own]) if own else np.zeros(0)
        own_m = sum(x[9] for x in own)
        rawd = np.array([x[10] for x in own if x[10] >= 0])
        fnear = np.array([x[11] for x in u]); fkept = np.array([x[12] for x in u])
        print('%-5s unm %4d (%.4f)  how %s | own: d<9 %d 9-12 %d >=12 %d, own matched-to-other %d, own raw_d<7 %d/%d | nearest preILP det <7um: %d (kept %d, dropped %d)' % (
            k, len(u), len(u) / n_all, dict(how), int((od < 9).sum()), int(((od >= 9) & (od < 12)).sum()), int((od >= 12).sum()), own_m,
            int((rawd < 7).sum()), len(rawd), int((fnear < 7).sum()), int(((fnear < 7) & (fkept == 1)).sum()), int(((fnear < 7) & (fkept == 0)).sum())))
        if own:
            o = np.array([x[6:9] for x in own])
            print('        own offset mean dz %+.2f dy %+.2f dx %+.2f  |dz| %.2f |dy| %.2f |dx| %.2f' % (o[:, 0].mean(), o[:, 1].mean(), o[:, 2].mean(), *np.abs(o).mean(0)))
