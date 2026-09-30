import json, glob, os, sys
import numpy as np
for grp in ['audit', 'hold36']:
    rows = json.load(open('/workspace/runs/divtest_%s/labeled.json' % grp))
    dn = {}
    for f in glob.glob('/workspace/runs/divnet_%s/*.divnet.json' % grp):
        m = os.path.basename(f).split('.')[0]
        dn.update({(m, int(k)): v for k, v in json.load(open(f)).items()})
    for r in rows: r['dn'] = dn.get((r['movie'], r['p']), np.nan)
    acc = lambda r: (r['typ'] == 'start' and r['fork'] >= .9) or (r['typ'] == 'stolen' and r['fork'] >= .97)
    print('==', grp, 'rows', len(rows), 'with dn', sum(1 for r in rows if r['dn'] == r['dn']))
    for lab in ['pos', 'neg', 'q_gt']:
        a = [r['dn'] for r in rows if r['lab'] == lab and acc(r)]
        print(' accepted', lab, len(a), 'dn', np.round(sorted(a), 3).tolist()[:40])
    pos = [(r['typ'], round(r['fork'], 3), round(r['dn'], 3)) for r in rows if r['lab'] == 'pos']
    print(' all pos (typ, fork, dn):', pos)
    for th in [0.3, 0.5, 0.7, 0.9]:
        c = {lab: sum(1 for r in rows if r['lab'] == lab and r['dn'] >= th) for lab in ['pos', 'neg', 'q_gt', 'unl']}
        print(' dn>=%.1f' % th, c)
