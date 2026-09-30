"""Approximate edge-count deltas (TP gained, FP/FN changes) of the deployed thresholds vs oracle, per set group.
relink accept: +1 TP if pos (else +1 FP if ev); removed current edge: if GT edge -> -1 TP (+1 FN); elif evaluable -> -1 FP.
edge_link accept: +1 TP if pos (gap1) / +2 TP if pos gap2 (approx), else +1 FP if ev. Greedy conflicts ignored (upper bound)."""
import json, glob
from collections import defaultdict
G = defaultdict(lambda: defaultdict(float))
clean = {'hold36', 'prev4'}
for f in glob.glob('/workspace/cl/lo2/*.json'):
    s, m = f.split('/')[-1][:-5].split('__'); grp = ('clean' if s in clean else 'train') + '_' + m[:4]
    d = json.load(open(f)); g = G[grp]
    for r in d.get('rl', []):
        dtp = (1 if r['pos'] else 0); dfp = (0 if r['pos'] else (1 if r['ev'] else 0))
        for ge, ev in [(r['cur_s_ge'], r['cur_s_ev']), (r['cur_d_ge'], r['cur_d_ev'])]:
            if ge == 1: dtp -= 1
            elif ev == 1: dfp -= 1
        g['rl_n'] += 1; g['rl_pos'] += r['pos']
        if r['p'] >= 0.65: g['rl_dep_dTP'] += dtp; g['rl_dep_dFP'] += dfp; g['rl_dep_n'] += 1
        if dtp > 0 or (dtp == 0 and dfp < 0): g['rl_or_dTP'] += dtp; g['rl_or_dFP'] += dfp; g['rl_or_n'] += 1
    for r in d.get('el', []):
        k = 1 if r['gap'] == 1 else 2
        dtp = k if r['pos'] else 0; dfp = 0 if r['pos'] else (1 if r['ev'] else 0)
        g['el_n'] += 1; g['el_pos'] += r['pos']
        if r['p'] >= 0.4: g['el_dep_dTP'] += dtp; g['el_dep_dFP'] += dfp; g['el_dep_n'] += 1
        if r['pos']: g['el_or_dTP'] += dtp; g['el_or_n'] += 1
for grp in sorted(G):
    print(grp, {k: int(v) for k, v in sorted(G[grp].items())})
