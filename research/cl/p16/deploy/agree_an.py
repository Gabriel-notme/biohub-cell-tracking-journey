"""Two-model agreement on division-completion candidates that P15 rejects (first dc pass, dnB logs).
fork_b1 = deployed b1, fork_new = DivNet trained on the OTHER embryo (LOEO), on b1's top-300 per type.
Pre-registered primary rule (fixed before looking): add if b1 >= 0.5 (but below P15's threshold) AND DivNet >= 0.9.
Positive = P or Ftp; negative = N, X or Ffp (official-style labels from nm/dv_cand). Break-even precision ~27%."""
import json, glob, os
from collections import defaultdict
cl = set(open('/workspace/hold36.txt').read().split()) | set(open('/workspace/preview4.txt').read().split())
th = {'start': 0.9, 'stolen': 0.97}
C = defaultdict(lambda: [0, 0, 0])  # key -> [TP, FP, unlabelled]
for f in glob.glob('/workspace/cl/p16/divnet/ps_dnB_*/dnlog/*_1.json'):
    name = os.path.basename(f).rsplit('_', 1)[0]; st = f.split('/ps_dnB_')[1].split('/')[0]
    lab = {}
    for r in json.load(open('/workspace/cl/nm/dv_cand/%s__%s.json' % (st, name))):
        if r['src'] == 'cand': lab[(r['p'], r['a'], r['b'])] = r['lab'][0]
        else: lab[(r['p'], r['a'], r['b'])] = r['lab']; lab[(r['p'], r['b'], r['a'])] = r['lab']
    for c in json.load(open(f)):
        b1, dn = c['fork_b1'], c['fork_new']
        if b1 >= th[c['typ']] or dn < 0: continue
        L = lab.get((c['p'], c['a'], c['b']), '?')
        y = 1 if L in ('P', 'Ftp') else (0 if L in ('N', 'X', 'Ffp') else -1)
        for bmin in (0.3, 0.5, 0.7):
            for dmin in (0.5, 0.7, 0.9, 0.97):
                if b1 >= bmin and dn >= dmin:
                    for grp in (name[:4], name[:4] + (':clean40' if name in cl else ':insample')):
                        k = (bmin, dmin, grp, c['typ'])
                        C[k][0 if y == 1 else (1 if y == 0 else 2)] += 1
print('PRIMARY (b1>=0.5 & DivNet>=0.9):')
for grp in ('44b6', '6bba', '44b6:clean40', '6bba:clean40', '44b6:insample', '6bba:insample'):
    for typ in ('start', 'stolen'):
        print('  %-14s %-6s TP %d FP %d unlabelled %d' % ((grp, typ) + tuple(C[(0.5, 0.9, grp, typ)])))
print('GRID (both types, both embryos): bmin dmin -> TP/FP (44b6) TP/FP (6bba)')
for bmin in (0.3, 0.5, 0.7):
    for dmin in (0.5, 0.7, 0.9, 0.97):
        a = [sum(C[(bmin, dmin, e, t)][i] for t in ('start', 'stolen')) for e in ('44b6', '6bba') for i in (0, 1)]
        print('  %.1f %.2f  44b6 %d/%d  6bba %d/%d' % (bmin, dmin, *a))
