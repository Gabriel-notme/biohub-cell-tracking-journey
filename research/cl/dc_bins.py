"""Division-completion candidates (first dc pass on the B5 graph, fork prob from the P-stage dump) joined with GT labels (cl2):
label counts per fork-probability bin, per set. b1 was trained on t127a/t127b/audit32 (in-sample, optimistic); hold36 is its calibration/audit split."""
import json, glob
from pathlib import Path
from collections import Counter
BINS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1.01]
for s in ['t127a', 't127b', 'audit32', 'hold36', 'prev4']:
    rows = []
    for f in glob.glob('/workspace/cl/cands_p8/%s/*.json' % s):
        m = Path(f).stem; lf = Path('/workspace/cl/cl2/%s/%s.json' % (s, m))
        if not lf.exists(): continue
        lab = {(r['p'], r['a'], r['b']): r['lab'] for r in json.load(open(lf))}
        for c in json.load(open(f)):
            if c['fork'] < 0.5: continue
            rows.append((c['typ'], c['fork'], lab.get((c['p'], c['a'], c['b']), 'X')))
    print(s)
    for typ in ['start', 'stolen']:
        line = '  %-6s' % typ
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            c = Counter(l for t, p, l in rows if t == typ and lo <= p < hi)
            line += ' [%.2f,%.2f) P%d D%d N%d |' % (lo, hi, c['P'], c['D'], c['N'])
        print(line)
