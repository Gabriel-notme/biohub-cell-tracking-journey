import sys, json, glob
from collections import Counter, defaultdict
sys.path.insert(0, '/workspace/official/src')
import warnings; warnings.filterwarnings('ignore')
from tracking_cellmot.metrics import summarise
KEYS = ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes']
D = {}
for f in sorted(glob.glob('/workspace/cl/p16/check2/placebo/char/*.json')):
    d = json.load(open(f)); D[d['info']['movie']] = d
ms = sorted(D); print('movies', len(ms))
print('eq_deployed', sum(D[m]['info']['eq_deployed'] for m in ms), 'exact', sum(D[m]['info']['exact'] for m in ms), 'edges_subset', sum(D[m]['info']['edges_subset'] for m in ms))
print('R0 nodes total', sum(D[m]['info']['R0'] for m in ms), 'of', sum(D[m]['info']['n15'] for m in ms))
claim = {r['movie']: r for r in json.load(open('/workspace/cl/rule_eval_last_ideas_p19r.json')) if r['vi'] == 1}
cb = {r['movie']: r for r in json.load(open('/workspace/cl/rule_eval_last_ideas_p19r.json')) if r['vi'] == 0}
print('claimed rows vi1 == my p19_full:', sum(all(claim[m][k] == D[m]['rows']['p19_full'][k] for k in KEYS) for m in ms),
      '| vi0 == my p15_full:', sum(all(cb[m][k] == D[m]['rows']['p15_full'][k] for k in KEYS) for m in ms))
G = [('all', ms), ('44b6', [m for m in ms if m.startswith('44b6')]), ('6bba', [m for m in ms if m.startswith('6bba')]),
     ('clean40', [m for m in ms if D[m]['info']['set'] in ('hold36', 'prev4')])]
for fam in ['cd', 'ff', 'st', 'tt', 'par', 'bd']:
    P = [p for m in ms for p in D[m]['info']['pieces'][fam]]
    typ = Counter((p['typ'], p['forky']) for p in P); sz = Counter(p['size'] for p in P)
    rel = [p for p in P if p['inR0']]
    print('\n%s: nodes %d pieces %d types %s sizes %s | pieces in R0 %d (nodes %d, matchable nodes %d) | movies with removal %d' % (
        fam, sum(p['size'] for p in P), len(P), dict(typ), dict(sorted(sz.items())), len(rel), sum(p['size'] for p in rel), sum(p['nmatch'] for p in rel),
        sum(1 for m in ms if D[m]['info']['rm'][fam])))
    line = ''
    for g, sel in G:
        a = summarise([D[m]['rows']['p15_red'] for m in sel]); b = summarise([D[m]['rows']['fam_' + fam] for m in sel])
        line += ' %s %+.6f' % (g, b['score'] - a['score'])
    dc = {k: sum(D[m]['rows']['fam_' + fam][k] - D[m]['rows']['p15_red'][k] for m in ms) for k in KEYS}
    print('  alone on P15:' + line, '| counts', dc)
line = ''
for g, sel in G:
    a = summarise([D[m]['rows']['p15_red'] for m in sel]); b = summarise([D[m]['rows']['p19_red'] for m in sel])
    line += ' %s %+.6f' % (g, b['score'] - a['score'])
print('\nP19-R (all families) vs P15:' + line, '| counts', {k: sum(D[m]['rows']['p19_red'][k] - D[m]['rows']['p15_red'][k] for k in KEYS) for m in [None]} if False else '')
print('  counts', {k: sum(D[m]['rows']['p19_red'][k] - D[m]['rows']['p15_red'][k] for m in ms) for k in KEYS})
print('timing mean', {k: sum(D[m]['info']['t'][k] for m in ms) / len(ms) for k in D[ms[0]]['info']['t']})
