"""check2/holdout (c): census of every P19-R (default) deleted item that touches an annotated GT track, and 10 random ones per family.
An item = connected piece of nodes deleted by one sub-rule (in the graph that sub-rule saw). 'touch' = one of its nodes is matched
to a GT node, or one of its incident P15 edges is TP or (valid) FP. marg = official counts change of deleting ONLY that item from P15."""
import json, glob, sys
import numpy as np
from collections import defaultdict, Counter

rowsdir = sys.argv[1]
FAM = {'cd': 'cutdup', 'ff': 'forkfrag', 'st': 'dup', 'tt': 'dup', 'par': 'dup', 'bd': 'border'}
items = []; nitems = Counter(); nnodes = Counter()
for f in sorted(glob.glob(rowsdir + '/*.json')):
    d = json.load(open(f))
    for it in d.get('items', []):
        it['movie'] = d['movie']; it['set'] = d['set']; nitems[it['fam']] += 1; nnodes[it['fam']] += len(it['nodes'])
        if it['touch']: items.append(it)


def verdict(it):
    m = it['marg']; dtp, dfp = m['edge_tp'], m['edge_fp']
    if m['division_tp'] < 0: return 'HARM-div'
    if dtp < 0 and dfp >= 0: return 'HARM'
    if dtp < 0: return 'MIXED(-tp,-fp)'
    if dfp < 0 or dtp > 0: return 'GOOD'
    if m['division_fp'] < 0: return 'GOOD-div'
    return 'neutral'


print('== census of deleted items (default P19-R on P15, 199 movies)')
for sf in ['cd', 'ff', 'st', 'tt', 'par', 'bd']:
    its = [it for it in items if it['fam'] == sf]
    v = Counter(verdict(it) for it in its)
    s = {k: sum(it['marg'][k] for it in its) for k in ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp']}
    print('  %-4s items %5d nodes %6d | touching annotated %3d (%.1f%%) | verdicts %s | summed single-item marg %s' % (
        sf, nitems[sf], nnodes[sf], len(its), 100. * len(its) / max(nitems[sf], 1), dict(v), s))
rng = np.random.default_rng(2026)
for fam in ['cutdup', 'forkfrag', 'dup', 'border']:
    its = [it for it in items if FAM[it['fam']] == fam]
    pick = [its[i] for i in sorted(rng.choice(len(its), min(10, len(its)), replace=False))] if its else []
    print('\n== %s: %d touching items, 10 random:' % (fam, len(its)))
    for it in pick:
        m = it['marg']
        ts = [p[0] for p in it['pos']]
        rem = it.get('rematch', {})
        print('  %-14s %-7s %-3s n=%d t%d-%d | matched %d (re-matched to another pred node after deletion: %d) | incident TP %d FP %d | marg tp %+d fp %+d fn %+d div %+d/%+d | nn_um min %.1f | %s' % (
            it['movie'], it['set'], it['fam'], len(it['nodes']), min(ts), max(ts), len(it['matched']), sum(1 for v in rem.values() if v is not None),
            len(it['tp_inc']), len(it['fp_inc']), m['edge_tp'], m['edge_fp'], m['edge_fn'], m['division_tp'], m['division_fp'], min(it['nn_um']), verdict(it)))
        print('      nodes %s pos(t,z,y,x) %s' % (it['nodes'][:6], it['pos'][:3]))
json.dump(items, open(rowsdir.rstrip('/') + '_touch_items.json', 'w'))
