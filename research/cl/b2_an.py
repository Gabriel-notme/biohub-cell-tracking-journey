import json, glob
import numpy as np
from pathlib import Path
R = []
for f in glob.glob('/workspace/cl/b2f/*.json'):
    s, m = Path(f).stem.split('__')
    b2 = json.load(open(f))
    if not b2: continue
    rows = json.load(open('/workspace/cl/cands/%s__%s.json' % (s, m))); labs = json.load(open('/workspace/cl/clab/%s__%s.json' % (s, m)))
    for i, p in b2.items():
        r = rows[int(i)]; R.append((s, r['typ'], labs[int(i)], r['fork'], p))
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
for typ in ['start', 'stolen']:
    for grp in [('t127a', 't127b', 'audit32'), ('hold36', 'prev4')]:
        q = [x for x in R if x[1] == typ and x[0] in grp and x[2] != 'U']
        if not q: continue
        lab = np.array([x[2] for x in q]); b1 = np.array([x[3] for x in q]); b2 = np.array([x[4] for x in q])
        mean = 1 / (1 + np.exp(-(lg(b1) + lg(b2)) / 2)); mn = np.minimum(b1, b2)
        print('== %s %s  P %d D %d N %d' % (typ, '+'.join(grp), (lab == 'P').sum(), (lab == 'D').sum(), (lab == 'N').sum()))
        for nm, sc in [('b1', b1), ('b2', b2), ('mean', mean), ('min', mn)]:
            line = '  %-5s' % nm
            for th in [0.5, 0.8, 0.9, 0.95, 0.97, 0.99]:
                k = sc >= th; line += ' | >=%.2f P%d D%d N%d' % (th, (k & (lab == 'P')).sum(), (k & (lab == 'D')).sum(), (k & (lab == 'N')).sum())
            y = (lab == 'P').astype(int)
            o = np.argsort(sc); rr = np.empty(len(sc)); rr[o] = np.arange(len(sc)); n1 = y.sum(); n0 = len(y) - n1
            auc = (rr[y == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)
            print(line + ' | AUC %.3f' % auc)
