import json, sys, glob
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
sys.path.insert(0, '/workspace/p56stage')
import jprune
cfg = json.load(open('/workspace/p56stage/p9_config.json'))['junk_prune']
tot = Counter()
for s in sys.argv[1:]:
    d8 = Path('/workspace/cl/ps_p8_%s/graphs' % s); d9 = Path('/workspace/cl/ps_p9_%s/graphs' % s)
    names8 = sorted(p.stem for p in d8.glob('*.json')); names9 = sorted(p.stem for p in d9.glob('*.json'))
    lst = [l.strip() for l in open('/workspace/%s.txt' % ('preview4' if s == 'prev4' else s)) if l.strip()]
    print('SET', s, 'n8', len(names8), 'n9', len(names9), 'list', len(lst), 'same names', names8 == names9, 'list==names', sorted(lst) == names8)
    for name in names8:
        g8 = json.load(open(d8 / (name + '.json'))); g9 = json.load(open(d9 / (name + '.json')))
        N8 = {int(k): v for k, v in g8['nodes'].items()}; N9 = {int(k): v for k, v in g9['nodes'].items()}
        E8 = [(int(e['source_id']), int(e['target_id'])) for e in g8['edges']]; E9 = [(int(e['source_id']), int(e['target_id'])) for e in g9['edges']]
        s8, s9 = set(E8), set(E9)
        probs = []
        if not set(N9) <= set(N8): probs.append('nodes not subset')
        if any(N9[n] != N8[n] for n in N9 if n in N8): probs.append('node attrs changed')
        if not s9 <= s8: probs.append('edges not subset')
        e8d = {(int(e['source_id']), int(e['target_id'])): e for e in g8['edges']}
        if any(e != e8d.get((int(e['source_id']), int(e['target_id']))) for e in g9['edges']): probs.append('edge attrs changed')
        R = set(N8) - set(N9)
        ch = defaultdict(list); par = defaultdict(list)
        for a, b in E8: ch[a].append(b); par[b].append(a)
        if max((len(v) for v in par.values()), default=0) > 1: probs.append('P8 has merges')
        exp9 = {(a, b) for a, b in s8 if a not in R and b not in R}
        if exp9 != s9: probs.append('edge set != P8 minus edges touching removed (%d vs %d)' % (len(exp9), len(s9)))
        adj = defaultdict(set)
        for a, b in E8: adj[a].add(b); adj[b].add(a)
        seen = set(); ncomp = 0; clens = []
        for r in R:
            if r in seen: continue
            comp = {r}; st = [r]
            while st:
                x = st.pop()
                for y in adj[x]:
                    if y not in comp: comp.add(y); st.append(y)
            seen |= comp; ncomp += 1; clens.append(len(comp))
            if not comp <= R: probs.append('partial component removed')
            if any(len(ch[x]) > 1 for x in comp): probs.append('removed comp has fork')
            if any(par[x] and len(ch[par[x][0]]) > 1 for x in comp): probs.append('removed node child of fork')
            roots = [x for x in comp if not par[x]]
            if len(roots) != 1: probs.append('comp roots %d' % len(roots))
        forks8 = {a: tuple(sorted(ch[a])) for a in ch if len(ch[a]) == 2}
        ch9 = defaultdict(list)
        for a, b in E9: ch9[a].append(b)
        forks9 = {a: tuple(sorted(ch9[a])) for a in ch9 if len(ch9[a]) == 2}
        if forks8 != forks9: probs.append('fork set differs')
        f8 = {int(v['t']) for v in N8.values()}; f9 = {int(v['t']) for v in N9.values()}
        if f8 != f9: probs.append('frames differ')
        if f9 != set(range(max(f9) + 1)): probs.append('frames not contiguous 0..T-1')
        ps = g9.get('pstage', {})
        if 'junk_prune_error' in ps or 'error' in ps: probs.append('pstage error ' + str({k: v for k, v in ps.items() if 'error' in k}))
        if ps.get('jp_removed_nodes') != len(R): probs.append('report mismatch %s vs %d' % (ps.get('jp_removed_nodes'), len(R)))
        n13, e13, st13 = jprune.apply(N8, g8['edges'], '/workspace/p56stage/jp_lgb.json', lam=cfg['lam'], tp_ref=cfg['tp_ref'], ratio=cfg['ratio'])
        if set(n13) != set(N9) or {(int(e['source_id']), int(e['target_id'])) for e in e13} != s9: probs.append('jprune(P8) != P9')
        c8 = Counter(int(v['t']) for v in N8.values()); c9 = Counter(int(v['t']) for v in N9.values())
        minfrac = min(c9[t] / c8[t] for t in c8)
        maxrm_t = max(c8[t] - c9[t] for t in c8)
        tot['movies'] += 1; tot['removed'] += len(R); tot['comps'] += ncomp; tot['n8'] += len(N8); tot['problems'] += bool(probs)
        print('%-16s n8 %6d removed %5d (%.2f%%) comps %4d maxcomplen %3d frames %3d minfrac_per_frame %.3f max_removed_in_frame %d %s' % (
            name, len(N8), len(R), 100 * len(R) / len(N8), ncomp, max(clens, default=0), len(f8), minfrac, maxrm_t,
            ('PROBLEMS ' + ';'.join(sorted(set(probs)))) if probs else 'OK'), flush=True)
print(dict(tot))
