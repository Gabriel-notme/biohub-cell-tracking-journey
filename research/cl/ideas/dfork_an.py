"""Read-only: for each dfork resolution in P13 (reconstructed), the kept later fork (label from div_lens_rows) and
whether a GT division lies near the EARLIER fork (|dt|<=1, <7um) or near the LATER fork. Also origin (dc/b5) of both forks."""
import os, sys, json
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
from collections import defaultdict, Counter
import numpy as np
import zarr
S = np.array([1.625, 0.40625, 0.40625])
A = json.load(open('/workspace/cl/ideas/div_lens_rows.json')); B = json.load(open('/workspace/cl/ideas/div_lens2_rows.json'))
lab = {(r['movie'], r['p']): r for r in A if r['kind'] == 'fork'}
D = [r for r in B if r['kind'] == 'dfork']
bym = defaultdict(list)
for r in D: bym[(r['set'], r['movie'])].append(r)
out = []
for (s, name), evs in bym.items():
    d = json.load(open('/workspace/cl/ps_p13_%s/graphs/%s.json' % (s, name)))
    nodes = {int(k): v for k, v in d['nodes'].items()}
    ch = defaultdict(list); par = {}
    for e in d['edges']: u, v = int(e['source_id']), int(e['target_id']); ch[u].append(v); par[v] = u
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}; t = {n: int(v['t']) for n, v in nodes.items()}
    z = zarr.open_group('/workspace/data/train/%s.geff' % name, mode='r')
    gids = np.asarray(z['nodes/ids']).tolist(); GP = np.stack([np.asarray(z['nodes/props/%s/values' % k]) for k in 'zyx'], 1) * S
    GT_ = np.asarray(z['nodes/props/t/values']).tolist(); GE = np.asarray(z['edges/ids']).tolist()
    gch = defaultdict(list)
    for u, v in GE: gch[u].append(v)
    gdiv = [(g, GT_[i], GP[i]) for i, g in enumerate(gids) if len(gch.get(g, [])) >= 2]
    gall = list(zip(gids, GT_, GP))

    def near_gdiv(n):
        return [g for g, tt, gp in gdiv if abs(tt - t[n]) <= 1 and np.linalg.norm(gp - pos[n]) < 7]

    def near_gt(n):
        return any(tt == t[n] and np.linalg.norm(gp - pos[n]) < 7 for g, tt, gp in gall)
    for r in evs:
        p = r['p']
        # later fork along kept branch
        later = None; x = r['kept'][0] if r['kept'] else None; k = 1
        while x is not None and k <= 36:
            if len(ch.get(x, [])) >= 2: later = x; break
            x = ch[x][0] if ch.get(x) else None; k += 1
        lr = lab.get((name, later)) if later is not None else None
        out.append(dict(set=s, movie=name, t=t[p], dt_later=(t[later] - t[p]) if later is not None else None,
                        later_lab=lr['lab'] if lr else None, later_origin=lr['origin'] if lr else None,
                        early_gdiv=len(near_gdiv(p)) > 0, early_annot=near_gt(p), later_gdiv=(len(near_gdiv(later)) > 0) if later is not None else None,
                        early_dc=r['dc'], lost_is_dc=[dc for v, dc in zip(r['kids'], r['dc']) if v in r['lost']]))
print('events', len(out))
print('later fork lab', Counter(o['later_lab'] for o in out))
print('early annotated (GT node within 7um at t):', Counter(o['early_annot'] for o in out))
print('early near GT div:', Counter((o['early_gdiv'], o['later_lab'], o['later_gdiv']) for o in out))
for o in out:
    if o['early_gdiv'] or o['later_lab'] in ('TP', 'FP') or o['early_annot']:
        print(o)
