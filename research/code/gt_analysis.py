import json, os, sys, glob
import numpy as np, zarr
from collections import Counter, defaultdict
root = '/workspace/data/train'
rows = []
for g in sorted(glob.glob(root + '/*.geff')):
    name = os.path.basename(g)[:-5]
    grp = zarr.open_group(g, mode='r')
    attrs = dict(grp.attrs)
    meta = attrs.get('geff', attrs)
    extra = meta.get('extra', {}) if isinstance(meta, dict) else {}
    est = extra.get('estimated_number_of_nodes')
    nid = np.asarray(grp['nodes/ids'][:])
    props = {k: np.asarray(grp[f'nodes/props/{k}/values'][:]) for k in ['t', 'z', 'y', 'x']}
    e = np.asarray(grp['edges/ids'][:]) if 'edges/ids' in grp else np.zeros((0, 2), int)
    idx = {int(n): i for i, n in enumerate(nid)}
    outdeg = Counter(int(s) for s, d in e); indeg = Counter(int(d) for s, d in e)
    t = props['t'].astype(int)
    ndiv = sum(1 for n, c in outdeg.items() if c >= 2)
    roots = [int(n) for n in nid if indeg.get(int(n), 0) == 0]
    leaves = [int(n) for n in nid if outdeg.get(int(n), 0) == 0]
    starts_t = Counter(int(t[idx[r]]) for r in roots)
    ends_t = Counter(int(t[idx[l]]) for l in leaves)
    per_t = Counter(t.tolist())
    rows.append(dict(movie=name, est=est, n=len(nid), e=len(e), div=ndiv, tmin=int(t.min()), tmax=int(t.max()),
                     frames=len(per_t), per_t_min=min(per_t.values()), per_t_max=max(per_t.values()),
                     roots=len(roots), roots_t0=starts_t.get(int(t.min()), 0), leaves=len(leaves),
                     leaves_tmax=ends_t.get(int(t.max()), 0),
                     z=(float(props['z'].min()), float(props['z'].max())), y=(float(props['y'].min()), float(props['y'].max())),
                     x=(float(props['x'].min()), float(props['x'].max())), extra_keys=list(extra.keys())))
json.dump(rows, open('/workspace/gt_summary.json', 'w'))
print(len(rows))
import statistics as st
for r in rows[:10]: print(r)
print('est ratio n/est median', st.median([r['n'] / r['est'] for r in rows if r['est']]))
print('frames', Counter(r['frames'] for r in rows).most_common(8))
print('tmin', Counter(r['tmin'] for r in rows).most_common(5), 'tmax', Counter(r['tmax'] for r in rows).most_common(5))
print('roots at t0 frac', st.mean([r['roots_t0'] / r['roots'] for r in rows]), 'leaves at tmax frac', st.mean([r['leaves_tmax'] / r['leaves'] for r in rows]))
print('total div', sum(r['div'] for r in rows), 'total edges', sum(r['e'] for r in rows))
