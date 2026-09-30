"""GT: frames between a division and the next division in either daughter branch (all annotated train movies)."""
import glob
import numpy as np
import zarr
from collections import defaultdict, Counter
gaps = []; nd = 0
for g in sorted(glob.glob('/workspace/data/train/*.geff')):
    z = zarr.open_group(g, mode='r')
    ids = np.asarray(z['nodes/ids']); T = np.asarray(z['nodes/props/t/values']); E = np.asarray(z['edges/ids'])
    t = dict(zip(ids.tolist(), T.tolist())); ch = defaultdict(list)
    for a, b in E.tolist(): ch[a].append(b)
    for d in [n for n in ch if len(ch[n]) >= 2]:
        nd += 1
        for c in ch[d]:
            x = c
            while len(ch.get(x, [])) == 1: x = ch[x][0]
            if len(ch.get(x, [])) >= 2: gaps.append(t[x] - t[d])
gaps = np.array(gaps)
print('GT divisions', nd, 'daughter branches that divide again', len(gaps))
print('min gap', gaps.min() if len(gaps) else None, 'quantiles 1/5/10/50%', np.percentile(gaps, [1, 5, 10, 50]) if len(gaps) else None)
print('gap histogram (<=30):', sorted(Counter(int(x) for x in gaps if x <= 30).items()))
