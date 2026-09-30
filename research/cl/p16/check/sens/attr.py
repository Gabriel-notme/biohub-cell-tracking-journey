import sys, json
import numpy as np
sys.path.insert(0, '/workspace/official/src')
import warnings; warnings.filterwarnings('ignore')
from tracking_cellmot.metrics import summarise
R = json.load(open('rows_main.json')); V = json.load(open('variants_main.json'))
by = {}
for r in R: by.setdefault(r['vi'], {})[r['movie']] = r
base = by[0]; ms = sorted(base)
S = lambda rows: summarise(rows)['score']
def d(cur, ref, sel): return S([cur[m] for m in sel]) - S([ref[m] for m in sel])
DK = ['division_tp', 'division_fp', 'division_fn']
for vi in range(1, len(V)):
    cur = by[vi]
    dm = [(m, base[m]['set'], tuple(cur[m][k] - base[m][k] for k in DK)) for m in ms if any(cur[m][k] != base[m][k] for k in DK)]
    # counterfactual: variant rows with base division counts -> non-division part
    nd = {m: dict(cur[m], **{k: base[m][k] for k in DK}) for m in ms}
    nodiv = d(nd, base, ms); ex = [m for m in ms if m not in {x[0] for x in dm}]
    print('%-16s all %+.5f  nondiv-part %+.5f  div-part %+.5f | excl div-changed movies (%d) %+.5f | div events: %s' % (
        json.dumps(V[vi]).replace(' ', ''), d(cur, base, ms), nodiv, d(cur, base, ms) - nodiv, len(dm), d(cur, base, ex), dm))
# per-set marginal of shortbranch (P17 vs sb off) and per-set P17
print()
sets = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
for lab, a, b in [('P17 vs P15', 1, 0), ('sb marginal (P17 vs sb0)', 1, 10), ('cd marginal', 1, 8), ('ff marginal', 1, 9), ('maxk3 vs P17', 7, 1), ('rad4 vs P17', 3, 1)]:
    line = '%-26s' % lab
    for s_ in sets:
        sel = [m for m in ms if base[m]['set'] == s_]; line += ' %s %+.5f' % (s_, d(by[a], by[b], sel))
    print(line)
