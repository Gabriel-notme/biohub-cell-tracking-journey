"""chk2 placebo (info): how many st/tt/par pieces are adjacent in P15 to nodes removed by earlier families (cd/ff), i.e. only exist
because an earlier family cut the track."""
import os, sys, json, glob
sys.path.insert(0, '/workspace/cl/p16/check2/placebo')
from lib import *
from multiprocessing import Pool


def job(args):
    s, f = args
    name = os.path.basename(f)[:-5]
    n15, e15 = load(f); refp = B5[s] + '/working/reference_graphs/%s.json' % name
    steps, _, _ = apply_rules(n15, e15, refp)
    adj = defaultdict(set)
    for e in e15:
        a, b = int(e['source_id']), int(e['target_id']); adj[a].add(b); adj[b].add(a)
    rm = {fam: r for fam, _, _, r in steps}; out = {}
    earlier = set()
    for fam, a, b, r in steps:
        P = piece_types(a, b, r)
        nb = sum(1 for p in P if any(y in earlier for x in p['nodes'] for y in adj[x]))
        nbcd = sum(1 for p in P if any(y in rm['cd'] for x in p['nodes'] for y in adj[x]))
        # structure of the piece in P15 itself (not in the pre-step graph)
        t15 = piece_types(n15, e15, set(x for p in P for x in p['nodes']))
        out[fam] = {'pieces': len(P), 'adj_earlier': nb, 'adj_cd': nbcd, 'types_in_p15': dict(Counter(q['typ'] for q in t15))}
        earlier |= r
    return name, out


if __name__ == '__main__':
    jobs = [(s, f) for s in SETS for f in sorted(glob.glob(P15 % s + '/*.json'))]
    with Pool(6, maxtasksperchild=8) as p: res = p.map(job, jobs, chunksize=1)
    tot = defaultdict(Counter)
    for _, o in res:
        for fam, d in o.items():
            tot[fam]['pieces'] += d['pieces']; tot[fam]['adj_earlier'] += d['adj_earlier']; tot[fam]['adj_cd'] += d['adj_cd']
            for k, v in d['types_in_p15'].items(): tot[fam]['p15type_' + k] += v
    for fam in FAMS: print(fam, dict(tot[fam]))
