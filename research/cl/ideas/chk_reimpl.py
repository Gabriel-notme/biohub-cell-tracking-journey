"""Independent re-implementation (check key 'reimpl') of the three P17 deletion rules, written ONLY from the textual description
(p17_post.py / p17_rules.py / r4_gapre.py / r4_forkfrag.py NOT read).  rule_eval-compatible: apply(nodes, edges, **kw) -> (nodes, edges, stats).
  cutdup     : surviving B5 gap bridges (edges flagged gap_closed / gap2_recovered); if a synthetic node of the bridge lies within thr um
               (anisotropic scale z 1.625, yx 0.40625) of another node of its frame, drop the whole synthetic chain.
  forkfrag   : weakly connected components with < ff_max nodes and no fork that contain a node which was a fork daughter in the B5
               reference graph and now has no parent -> removed.
  shortbranch: fork whose daughter branch ends within 2 frames (<= sb_max nodes, no further fork) -> delete that branch (one per fork).
"""
import json
from collections import defaultdict
import numpy as np

WANTS_META = True
REF = {'hold36': '/workspace/runs/b5f_hold36/working/reference_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/reference_graphs',
       'audit32': '/workspace/sync3/runs/b5f_audit32/working/reference_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/reference_graphs',
       't127b': '/workspace/sync3/runs/b5f_t127b/working/reference_graphs'}
SC = np.array([1.625, 0.40625, 0.40625])


def _xyz(v):
    return np.array([v['z'], v['y'], v['x']], float) * SC


def _load_ref(set, name):
    d = json.load(open(REF[set] + '/' + name + '.json'))
    return {int(k): v for k, v in d['nodes'].items()}, d['edges']


def _remove(nodes, edges, drop):
    if not drop: return nodes, edges
    nn = {k: v for k, v in nodes.items() if k not in drop}
    ne = [e for e in edges if int(e['source_id']) not in drop and int(e['target_id']) not in drop]
    return nn, ne


def _children(edges):
    ch = defaultdict(list); par = defaultdict(list)
    for e in edges:
        s, t = int(e['source_id']), int(e['target_id']); ch[s].append(t); par[t].append(s)
    return ch, par


def cutdup(nodes, edges, rn, re_, thr=3.2, surv='flag', mode='any', g2src='ref'):
    FL = ('gap_closed', 'gap2_recovered')
    flagged = [e for e in edges if any(e.get(k) for k in FL)]
    inc = defaultdict(int)
    for e in flagged: inc[int(e['source_id'])] += 1; inc[int(e['target_id'])] += 1
    syn = {n for n, v in nodes.items() if v.get('gap_synthetic')}
    if g2src == 'ref':  # interior nodes of gap2 chains in the B5 reference graph (the nodes gap2 recovery inserted)
        ri = defaultdict(int); ro = defaultdict(int)
        for e in re_:
            if e.get('gap2_recovered'): ro[int(e['source_id'])] += 1; ri[int(e['target_id'])] += 1
        g2 = {n for n in ri if ro.get(n) and n in nodes and nodes[n]['t'] == rn[n]['t']}
        # also the reference gap_synthetic nodes (in case the flag was lost downstream)
        syn |= {n for n, v in rn.items() if v.get('gap_synthetic') and n in nodes and nodes[n]['t'] == v['t']}
    else:  # local: interior of gap2-flagged chains in the current graph
        gi = defaultdict(int); go = defaultdict(int)
        for e in edges:
            if e.get('gap2_recovered'): go[int(e['source_id'])] += 1; gi[int(e['target_id'])] += 1
        g2 = {n for n in gi if go.get(n)}
    syn |= g2
    if surv == 'flag':
        cand = {n for n in syn if inc.get(n)}
    elif surv == 'full':  # both bridge edges still present
        cand = {n for n in syn if inc.get(n, 0) >= 2}
    else:
        cand = set(syn)
    if not cand: return set(), 0
    # chains: synthetic nodes joined by flagged edges
    parent = {n: n for n in cand}

    def f(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for e in flagged:
        s, t = int(e['source_id']), int(e['target_id'])
        if s in cand and t in cand: parent[f(s)] = f(t)
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[v['t']].append(n)
    arr = {t: (np.array(ids), np.array([_xyz(nodes[i]) for i in ids])) for t, ids in byt.items() if any(i in cand for i in ids)}
    close = {}
    for n in cand:
        ids, X = arr[nodes[n]['t']]
        d = np.sqrt(((X - _xyz(nodes[n])) ** 2).sum(1)); d[ids == n] = np.inf
        close[n] = bool(d.min() <= thr) if len(d) > 1 else False
    groups = defaultdict(list)
    for n in cand: groups[f(n)].append(n)
    drop = set(); nch = 0
    for g in groups.values():
        hit = any(close[n] for n in g) if mode == 'any' else all(close[n] for n in g)
        if hit: drop |= set(g); nch += 1
    return drop, nch


def forkfrag(nodes, edges, rn, re_, ff_max=6):
    rch, _ = _children(re_)
    daughters = {c for p, cs in rch.items() if len(cs) >= 2 for c in cs}
    ch, par = _children(edges)
    orph = {n for n in daughters if n in nodes and nodes[n]['t'] == rn[n]['t'] and not par.get(n)}
    if not orph: return set(), 0
    parent = {n: n for n in nodes}

    def f(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for e in edges: parent[f(int(e['source_id']))] = f(int(e['target_id']))
    comp = defaultdict(list)
    for n in nodes: comp[f(n)].append(n)
    drop = set(); nc = 0
    for r in {f(n) for n in orph}:
        c = comp[r]
        if len(c) < ff_max and not any(len(ch.get(n, [])) >= 2 for n in c):
            drop |= set(c); nc += 1
    return drop, nc


def shortbranch(nodes, edges, sb_max=3, tie='short', skip_end=False):
    ch, par = _children(edges)
    tmax = max(v['t'] for v in nodes.values()) if nodes else 0
    drop = set(); nb = 0
    for p in sorted(ch):
        if len(ch[p]) < 2: continue
        shorts = []
        for c in ch[p]:
            br = [c]; n = c; ok = True
            while True:
                k = ch.get(n, [])
                if len(k) == 0: break
                if len(k) >= 2: ok = False; break
                n = k[0]; br.append(n)
                if len(br) > sb_max: ok = False; break
            if ok and len(br) <= sb_max:
                if skip_end and nodes[br[-1]]['t'] == tmax: continue
                shorts.append(br)
        if not shorts: continue
        if tie == 'short': shorts.sort(key=lambda b: (len(b), b[0]))
        elif tie == 'long': shorts.sort(key=lambda b: (-len(b), b[0]))
        elif tie == 'order': shorts.sort(key=len)  # stable: first child in edge-list order among the shortest
        drop |= set(shorts[0]); nb += 1
    return drop, nb


def apply(nodes, edges, rules=('cutdup', 'forkfrag', 'shortbranch'), thr=3.2, surv='flag', mode='any', g2src='ref', ff_max=6, sb_max=3,
          tie='short', skip_end=False, name=None, set=None, fullgeff=None, zarr=None):
    rn, re_ = _load_ref(set, name)
    st = {}
    for r in rules:
        if r == 'cutdup':
            drop, k = cutdup(nodes, edges, rn, re_, thr=thr, surv=surv, mode=mode, g2src=g2src)
        elif r == 'forkfrag':
            drop, k = forkfrag(nodes, edges, rn, re_, ff_max=ff_max)
        elif r == 'shortbranch':
            drop, k = shortbranch(nodes, edges, sb_max=sb_max, tie=tie, skip_end=skip_end)
        st['x_' + r] = k; st['xn_' + r] = len(drop)
        nodes, edges = _remove(nodes, edges, drop)
    return nodes, edges, st

