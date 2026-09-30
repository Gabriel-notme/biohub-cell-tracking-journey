import os
for k in ['OMP_NUM_THREADS', 'POLARS_MAX_THREADS', 'RAYON_NUM_THREADS']: os.environ[k] = '1'
import sys, csv, hashlib, json
from collections import Counter, defaultdict
paths = sys.argv[1:]
for p in paths:
    nodes = defaultdict(dict); edges = defaultdict(list); ids = set(); bad = Counter()
    with open(p) as f:
        for r in csv.DictReader(f):
            if r['id'] in ids: bad['dup_row_id'] += 1
            ids.add(r['id']); ds = r['dataset']
            if r['row_type'] == 'node':
                n = int(r['node_id'])
                if n in nodes[ds]: bad['dup_node'] += 1
                nodes[ds][n] = (int(r['t']), int(r['z']), int(r['y']), int(r['x']))
                if min(nodes[ds][n][1:]) < 0 or max(nodes[ds][n][2:]) > 255 or nodes[ds][n][1] > 63: bad['oob'] += 1
            elif r['row_type'] == 'edge':
                edges[ds].append((int(r['source_id']), int(r['target_id'])))
            else: bad['row_type'] += 1
    summ = []
    for ds in sorted(set(nodes) | set(edges)):
        N = nodes[ds]; E = edges[ds]
        if len(E) != len(set(E)): bad['dup_edge'] += 1
        if max(Counter(t for s, t in E).values(), default=0) > 1: bad['merge'] += 1
        if max(Counter(s for s, t in E).values(), default=0) > 2: bad['outdeg>2'] += 1
        for s, t in E:
            if s not in N or t not in N: bad['dangling'] += 1
            elif N[t][0] != N[s][0] + 1: bad['nonconsec'] += 1
        if {v[0] for v in N.values()} != set(range(max(v[0] for v in N.values()) + 1)): bad['frame_gap'] += 1
        summ.append((ds, len(N), len(E), sum(1 for c in Counter(s for s, t in E).values() if c == 2)))
    print(p, 'datasets', len(summ), 'problems', dict(bad) or 'none', 'sha256', hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16])
    for x in summ[:6]: print('  ', x)
