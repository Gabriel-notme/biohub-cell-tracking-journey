import csv, sys
from collections import Counter, defaultdict
def rd(p):
    N = defaultdict(dict); E = defaultdict(set)
    for r in csv.DictReader(open(p)):
        if r['row_type'] == 'node': N[r['dataset']][int(r['node_id'])] = (r['t'], r['z'], r['y'], r['x'])
        else: E[r['dataset']].add((int(r['source_id']), int(r['target_id'])))
    return N, E
def forks(E):
    o = defaultdict(list)
    for s, t in E: o[s].append(t)
    return {p: tuple(sorted(c)) for p, c in o.items() if len(c) == 2}
a, b = sys.argv[1], sys.argv[2]
Na, Ea = rd(a); Nb, Eb = rd(b)
for ds in sorted(Na):
    na, nb = Na[ds], Nb[ds]; ea, eb = Ea[ds], Eb[ds]
    fa, fb = forks(ea), forks(eb)
    print(ds, 'nodes %d->%d (del %d, new %d, moved %d)' % (len(na), len(nb), len(set(na) - set(nb)), len(set(nb) - set(na)), sum(1 for k in set(na) & set(nb) if na[k] != nb[k])),
          'edges %d->%d (del %d, new %d)' % (len(ea), len(eb), len(ea - eb), len(eb - ea)), 'forks %d->%d lost %d' % (len(fa), len(fb), sum(1 for p, c in fa.items() if fb.get(p) != c)))
