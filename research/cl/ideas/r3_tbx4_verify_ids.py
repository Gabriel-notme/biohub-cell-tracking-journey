"""Verifier: clean40 leave-one-movie-out, and independent id check (P14 node ids vs fullgraph ids). No GT read."""
import json, sys, glob
import numpy as np
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p56stage')
import warnings; warnings.filterwarnings('ignore')
from tracking_cellmot.metrics import summarise
R = json.load(open('/workspace/cl/ideas/r3_tbx4_verify_rows.json'))
base = {r['movie']: r for r in R if r['vi'] == 0}
ms = sorted(base)
cm = [m for m in ms if base[m]['set'] in ('hold36', 'prev4')]
for i, vn in enumerate(['p0.5', 'p0.5+join', 'p0.8', 'p0.8+join'], 1):
    cur = {r['movie']: r for r in R if r['vi'] == i}
    d = lambda mm: summarise([cur[m] for m in mm])['score'] - summarise([base[m] for m in mm])['score']
    full = d(cm)
    loo = sorted(((d([x for x in cm if x != m]), m) for m in cm))
    ce = {e: d([m for m in cm if m.startswith(e)]) for e in ['44b6', '6bba']}
    print('%s clean40 %+.5f (44b6 %+.5f, 6bba %+.5f) min LOO %+.5f (drop %s) max LOO %+.5f (drop %s); clean40 up/down %d/%d' % (
        vn, full, ce['44b6'], ce['6bba'], loo[0][0], loo[-1][1], loo[-1][0], loo[0][1],
        sum(cur[m]['adj_edge_jaccard'] > base[m]['adj_edge_jaccard'] for m in cm), sum(cur[m]['adj_edge_jaccard'] < base[m]['adj_edge_jaccard'] for m in cm)))
    # clean40 excluding the single dense prev4 movie
    print('   clean40 w/o 6bba_05db0fb1 %+.5f' % d([m for m in cm if m != '6bba_05db0fb1']))
# id check on a sample of movies
import evalx
from edge_link import load_full
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
S = np.array([1.625, 0.40625, 0.40625])
for s in FULL:
    fs = sorted(glob.glob('/workspace/cl/ps_p14_%s/graphs/*.json' % s))[:2]
    for f in fs:
        name = f.split('/')[-1][:-5]
        nodes, edges = evalx.load_graph_json(f)
        fids, fT, fV, fE, fprob = load_full(FULL[s] + '/' + name + '.geff')
        fidx = {int(i): j for j, i in enumerate(fids.tolist())}
        inf = [n for n in nodes if int(n) in fidx]
        tmis = sum(1 for n in inf if int(fT[fidx[int(n)]]) != int(nodes[n]['t']))
        dd = np.array([np.linalg.norm((np.array([nodes[n][k] for k in 'zyx'], float) - fV[fidx[int(n)]]) * S) for n in inf if int(fT[fidx[int(n)]]) == int(nodes[n]['t'])])
        print('%s %s: P14 nodes %d, in fullgraph %d (%.4f), t-mismatch %d, pos diff median %.2f um p99 %.2f um, full nodes %d' % (
            s, name, len(nodes), len(inf), len(inf) / len(nodes), tmis, np.median(dd), np.quantile(dd, .99), len(fids)))
