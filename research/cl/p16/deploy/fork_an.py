"""Structural anatomy of P15 forks: TP (Ftp) vs FP (Ffp) forks by official-style labels (nm/dv_cand), per embryo.
Looks for a GT-free structural criterion that removes many FP forks and few TP forks in BOTH embryos."""
import json, glob, os
from collections import defaultdict
import numpy as np
S = np.array([1.625, .40625, .40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
rows = []
for s in SETS:
    for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s)):
        name = os.path.basename(f)[:-5]
        lf = '/workspace/cl/nm/dv_cand/%s__%s.json' % (s, name)
        if not os.path.exists(lf): continue
        lab = {}
        for r in json.load(open(lf)):
            if r['src'] != 'cand': lab[r['p']] = r['lab']
        d = json.load(open(f)); nodes = {int(k): v for k, v in d['nodes'].items()}
        out = defaultdict(list); prev = {}; eattr = {}
        for e in d['edges']:
            a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); prev[b] = a; eattr[(a, b)] = e
        pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
        frames = defaultdict(list)
        for n, v in nodes.items(): frames[int(v['t'])].append(n)
        def fwd(n):  # frames until the branch ends or divides again; returns (length, ends, divides)
            k = 0
            while True:
                c = out.get(n, [])
                if len(c) == 0: return k, True, False
                if len(c) == 2: return k, False, True
                n = c[0]; k += 1
                if k >= 60: return k, False, False
        def back(n):  # frames since the track start or the previous fork
            k = 0
            while n in prev:
                q = prev[n]
                if len(out[q]) == 2: return k, True
                n = q; k += 1
                if k >= 200: break
            return k, False
        for p, ch in out.items():
            if len(ch) != 2 or p not in lab: continue
            L = lab[p]
            if L not in ('Ftp', 'Ffp'): continue
            a, b = ch
            fa, fb = fwd(a), fwd(b)
            ag, prevdiv = back(p)
            flags = set()
            for c in (a, b):
                e = eattr[(p, c)]
                for k in ('div_complete', 'safe_division', 'joint_event', 'relink', 'edge_link', 'gap_closed', 'gap2_recovered'):
                    if e.get(k): flags.add(k)
            t = int(nodes[p]['t'])
            dens = sum(1 for n in frames[t] if n != p and np.linalg.norm(pos[n] - pos[p]) < 8)
            rows.append(dict(emb=name[:4], set=s, movie=name, y=1 if L == 'Ftp' else 0, prov='dc' if 'div_complete' in flags else ('safe' if 'safe_division' in flags else ('joint' if 'joint_event' in flags else 'other')),
                             minlen=min(fa[0], fb[0]), maxlen=max(fa[0], fb[0]), end_short=int(min(fa[0], fb[0]) < 5 and (fa[1] if fa[0] < fb[0] else fb[1])),
                             redivide=min(fa[0] if fa[2] else 99, fb[0] if fb[2] else 99), age=ag, prevdiv=int(prevdiv),
                             sis=float(np.linalg.norm(pos[a] - pos[b])), dmax=float(max(np.linalg.norm(pos[a] - pos[p]), np.linalg.norm(pos[b] - pos[p]))),
                             dens=dens, t=t, edge=int(t < 3 or t > max(frames) - 4)))
json.dump(rows, open('/workspace/cl/p16/deploy/fork_an_rows.json', 'w'))
def show(label, sel):
    for e in ('44b6', '6bba'):
        R = [r for r in rows if r['emb'] == e]
        tp = [r for r in R if r['y'] == 1]; fp = [r for r in R if r['y'] == 0]
        st = sum(1 for r in tp if sel(r)); sf = sum(1 for r in fp if sel(r))
        print('  %-44s %s: removes TP %2d/%2d FP %2d/%2d  net %+.5f' % (label, e, st, len(tp), sf, len(fp), 0.00022 * sf - 0.00056 * st))
print('forks labelled:', len(rows), 'TP', sum(r['y'] for r in rows), 'FP', sum(1 - r['y'] for r in rows))
for k in ('prov',):
    for v in sorted({r[k] for r in rows}):
        print(k, v, 'TP', sum(1 for r in rows if r[k] == v and r['y']), 'FP', sum(1 for r in rows if r[k] == v and not r['y']))
for k in ('minlen', 'maxlen', 'redivide', 'age', 'sis', 'dmax', 'dens', 'end_short', 'prevdiv', 'edge'):
    tp = np.array([r[k] for r in rows if r['y']]); fp = np.array([r[k] for r in rows if not r['y']])
    print('%-9s TP median %.1f q10 %.1f q90 %.1f | FP median %.1f q10 %.1f q90 %.1f' % (k, np.median(tp), np.quantile(tp, .1), np.quantile(tp, .9), np.median(fp), np.quantile(fp, .1), np.quantile(fp, .9)))
for th in (3, 5, 8, 12): show('minlen < %d' % th, lambda r, th=th: r['minlen'] < th)
for th in (5, 10, 20, 35): show('redivide < %d' % th, lambda r, th=th: r['redivide'] < th)
for th in (5, 10, 20): show('age < %d & prevdiv' % th, lambda r, th=th: r['age'] < th and r['prevdiv'])
for th in (4, 6, 12, 16): show('sis > %d' % th, lambda r, th=th: r['sis'] > th)
show('end_short', lambda r: r['end_short'])
show('edge frames', lambda r: r['edge'])
