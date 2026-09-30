"""Per embryo: GT divisions, how many daughter branches are annotated for >= k frames after the division (the opportunity to see a
second division), and nested divisions found. Upper 95% bound on the nested-division rate (rule of three)."""
import glob
import numpy as np
import zarr
from collections import defaultdict
for emb in ['44b6', '6bba']:
    nd = 0; branches = defaultdict(int); nested = 0; movies = 0; movies_div = 0
    for g in sorted(glob.glob('/workspace/data/train/%s_*.geff' % emb)):
        z = zarr.open_group(g, mode='r'); movies += 1
        ids = np.asarray(z['nodes/ids']); T = np.asarray(z['nodes/props/t/values']); E = np.asarray(z['edges/ids'])
        t = dict(zip(ids.tolist(), T.tolist())); ch = defaultdict(list)
        for a, b in E.tolist(): ch[a].append(b)
        divs = [n for n in ch if len(ch[n]) >= 2]; nd += len(divs); movies_div += bool(divs)
        for d in divs:
            for c in ch[d]:
                x = c; L = 1
                while len(ch.get(x, [])) == 1: x = ch[x][0]; L += 1
                if len(ch.get(x, [])) >= 2: nested += 1
                for k in [10, 20, 40]:
                    if L >= k: branches[k] += 1
    print('%s: movies %d (with divisions %d), GT divisions %d, nested %d, daughter branches followed >=10/20/40 frames: %d/%d/%d, 95%% upper bound nested rate per 40-frame branch %.3f' % (
        emb, movies, movies_div, nd, nested, branches[10], branches[20], branches[40], 3 / max(1, branches[40])))
