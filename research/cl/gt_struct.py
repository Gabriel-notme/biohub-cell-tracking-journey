import zarr, numpy as np, glob
from collections import defaultdict, Counter
from pathlib import Path
names = sorted(Path(p).stem for p in glob.glob('/workspace/data/train/*.geff'))
comp_stats = Counter(); per = []
for n in names:
    g = zarr.open_group('/workspace/data/train/%s.geff' % n, mode='r')
    ids = np.asarray(g['nodes/ids'][:]); T = np.asarray(g['nodes/props/t/values'][:])
    E = np.asarray(g['edges/ids'][:])
    par = {}; succ = defaultdict(list)
    for s, d in E.tolist(): succ[s].append(d); par[d] = s
    # components
    adj = defaultdict(list)
    for s, d in E.tolist(): adj[s].append(d); adj[d].append(s)
    seen = set(); comps = []
    tmap = dict(zip(ids.tolist(), T.tolist()))
    for i in ids.tolist():
        if i in seen: continue
        st = [i]; seen.add(i); c = []
        while st:
            x = st.pop(); c.append(x)
            for y in adj[x]:
                if y not in seen: seen.add(y); st.append(y)
        comps.append(c)
    for c in comps:
        nd = sum(1 for x in c if len(succ.get(x, [])) >= 2)
        ts = [tmap[x] for x in c]
        comp_stats[('div' if nd else 'nodiv', 'len>=20' if len(c) >= 20 else 'short')] += 1
    per.append((n, len(ids), len(comps), sum(1 for x in ids.tolist() if len(succ.get(x, [])) >= 2)))
print(comp_stats)
a = np.array([p[1:] for p in per])
print('movies', len(per), 'nodes mean %.0f comps mean %.1f median %d divs total %d' % (a[:, 0].mean(), a[:, 1].mean(), np.median(a[:, 1]), a[:, 2].sum()))
print('comps distribution', np.percentile(a[:, 1], [10, 25, 50, 75, 90]))
print('movies with 0 div', (a[:, 2] == 0).sum(), '1 div', (a[:, 2] == 1).sum(), '2+', (a[:, 2] >= 2).sum())
