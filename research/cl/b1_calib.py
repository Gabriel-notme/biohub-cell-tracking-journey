"""Is the b1 fork head calibrated on movies it never saw? Division-completion candidates (fork prob from the P-stage dump on the B5
graph) joined with GT labels (P new division, D already recovered, N not a division). Compare b1-seen movies (t127a/t127b/audit32)
with b1-unseen movies (hold36 = b1 calibration/audit split, prev4 = never used), per embryo."""
import json, glob
from pathlib import Path
import numpy as np
from collections import defaultdict
rows = []
for s in ['t127a', 't127b', 'audit32', 'hold36', 'prev4']:
    for f in glob.glob('/workspace/cl/cands_p8/%s/*.json' % s):
        m = Path(f).stem; lf = Path('/workspace/cl/cl2/%s/%s.json' % (s, m))
        if not lf.exists(): continue
        lab = {(r['p'], r['a'], r['b']): r['lab'] for r in json.load(open(lf))}
        for c in json.load(open(f)):
            l = lab.get((c['p'], c['a'], c['b']))
            if l in ('P', 'D', 'N'): rows.append((s, m[:4], c['typ'], l, c['fork']))
seen = {'t127a', 't127b', 'audit32'}


def auc(pos, neg):
    if not pos or not neg: return float('nan')
    a = np.array(pos); b = np.array(neg)
    return float(((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


for grp, f in [('b1-seen (t127+audit32)', lambda s: s in seen), ('b1-unseen (hold36+prev4)', lambda s: s not in seen)]:
    for typ in ['start', 'stolen']:
        for emb in ['44b6', '6bba', 'all']:
            P = [p for s, e, t, l, p in rows if f(s) and t == typ and l in ('P', 'D') and (emb == 'all' or e == emb)]
            N = [p for s, e, t, l, p in rows if f(s) and t == typ and l == 'N' and (emb == 'all' or e == emb)]
            q = lambda v, x: np.percentile(v, x) if v else float('nan')
            print('%-25s %-6s %-4s  true n=%3d median %.3f p25 %.3f | neg n=%6d  frac>=0.9 %.4f | AUC %.3f  true>=th %s' % (
                grp, typ, emb, len(P), q(P, 50), q(P, 25), len(N), np.mean(np.array(N) >= 0.9) if N else float('nan'), auc(P, N),
                '%.2f' % np.mean(np.array(P) >= (0.9 if typ == 'start' else 0.97)) if P else 'nan'))
