"""Read-only: extra fork context on P13 graphs, joined with div_lens_rows labels.
(a) nearest other fork in space-time; (b) track ends at t / t-1 near each daughter (re-appearance), with history;
(c) dc type (start/stolen from B5 parent of b); (d) dfork events reconstructed (B5 + dc edges vs P13)."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}


def hist(x, par, lim=60):
    h = 0
    while x in par and h < lim: x = par[x]; h += 1
    return h


def job(a):
    s, f = a
    name = Path(f).stem
    d = json.loads(Path(f).read_text())
    nodes = {int(k): v for k, v in d['nodes'].items()}; edges = d['edges']
    b = json.loads(Path(B5[s] + '/' + name + '.json').read_text())
    bn = {int(k): v for k, v in b['nodes'].items()}; be = b['edges']
    bch = defaultdict(list); bpar = {}
    for e in be: u, v = int(e['source_id']), int(e['target_id']); bch[u].append(v); bpar[v] = u
    ch = defaultdict(list); par = {}; eattr = {}
    for e in edges:
        u, v = int(e['source_id']), int(e['target_id']); ch[u].append(v); par[v] = u; eattr[(u, v)] = e
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}; t = {n: int(v['t']) for n, v in nodes.items()}
    byt = defaultdict(list)
    for n in nodes: byt[t[n]].append(n)
    forks = [n for n in ch if len(ch[n]) >= 2]
    out = []
    for p in forks:
        kids = ch[p][:2]
        # (a) nearest other fork
        best = (99., 99)
        for q in forks:
            if q == p or abs(t[q] - t[p]) > 3: continue
            dd = float(np.linalg.norm(pos[q] - pos[p]))
            if dd < best[0]: best = (dd, t[q] - t[p])
        # (b) track ends near each daughter at t (gap1) and t-1 (gap2), excluding p and its lineage
        ends = []
        for k in kids:
            for dt_ in (0, 1):
                for r in byt[t[p] - dt_]:
                    if r == p or ch.get(r): continue
                    dd = float(np.linalg.norm(pos[r] - pos[k]))
                    if dd <= 6.0: ends.append(dict(kid=k, gap=dt_ + 1, d=round(dd, 2), dp=round(float(np.linalg.norm(pos[k] - pos[p])), 2), h=hist(r, par)))
        # (c) dc type
        dck = [k for k in kids if 'div_complete' in eattr[(p, k)]]
        dctype = None
        if dck:
            k = dck[0]; q = bpar.get(k)
            dctype = 'start' if q is None else ('stolen' if q != p else 'same')
        out.append(dict(kind='fork', set=s, movie=name, p=p, nf_d=best[0], nf_dt=best[1], ends=ends, dctype=dctype))
    # (d) dfork: forks in (B5 + dc additions - dc stolen removals) that are not forks in P13 and lost a child edge
    pre = {(int(e['source_id']), int(e['target_id'])) for e in be}
    for e in edges:
        if 'div_complete' in e:
            u, v = int(e['source_id']), int(e['target_id']); pre.add((u, v))
            q = bpar.get(v)
            if q is not None and q != u: pre.discard((q, v))
    pch = defaultdict(list)
    for u, v in pre: pch[u].append(v)
    post = {(int(e['source_id']), int(e['target_id'])) for e in edges}
    for u, vs in pch.items():
        if len(vs) < 2 or len(ch.get(u, [])) >= 2 or u not in nodes: continue
        lost = [v for v in vs if (u, v) not in post]
        out.append(dict(kind='dfork', set=s, movie=name, p=u, t=t[u], kids=vs, lost=lost, kept=[v for v in vs if (u, v) in post],
                        dc=[('div_complete' in eattr.get((u, v), {})) or ((u, v) not in {(int(e['source_id']), int(e['target_id'])) for e in be}) for v in vs]))
    return out


if __name__ == '__main__':
    jobs = [(s, f) for s in B5 for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(12) as p: R = [r for rs in p.map(job, jobs, chunksize=1) for r in rs]
    json.dump(R, open('/workspace/cl/ideas/div_lens2_rows.json', 'w'))
    print('rows', len(R))
