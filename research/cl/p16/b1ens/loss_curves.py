"""Training loss curves from train_events.py logs: mean loss per 1000-step block, per kind (edge / fork)."""
import sys, re, glob, json
import numpy as np
for f in sys.argv[1:]:
    L = {'edge': {}, 'fork': {}}; val = None
    for line in open(f):
        m = re.match(r'STEP (\d+) (edge|fork) loss ([0-9.e-]+)', line)
        if m: L[m.group(2)].setdefault((int(m.group(1)) - 1) // 1000, []).append(float(m.group(3)))
        if line.startswith('VALIDATION'): val = json.loads(line[11:])
    done = 'TRAIN_DONE' in open(f).read()
    print('%-26s done=%s' % (f.split('/')[-1][:-4], done))
    for k in ('edge', 'fork'):
        print('   %-4s ' % k + ' '.join('%.4f' % np.mean(L[k][b]) for b in sorted(L[k])))
    if val: print('   last val on split calibration movies (deploy: in-sample; LOEO/half: held-out same embryo) edge AP %.4f fork AP %.3f (pos %d) fork logloss %.4f step %d' % (val['edge']['ap'], val['fork']['ap'], val['fork']['positive'], val['fork']['logloss'], val['step']))
