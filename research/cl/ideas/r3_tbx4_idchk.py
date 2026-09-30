"""ID-consistency check: P14 node ids vs pre-ILP fullgraph ids (positions must agree), per set, a few movies each."""
import sys, glob, json
import numpy as np
sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/code')
from edge_link import load_full
import evalx
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
for s, fd in FULL.items():
    fs = sorted(glob.glob('/workspace/cl/ps_p14_%s/graphs/*.json' % s))[:3]
    for f in fs:
        name = f.split('/')[-1][:-5]
        nodes, edges = evalx.load_graph_json(f)
        fids, fT, fV, fE, fprob = load_full(fd + '/' + name + '.geff')
        fidx = {int(i): j for j, i in enumerate(fids.tolist())}
        inn = [n for n in nodes if n in fidx]
        dpos = [np.abs(np.array([nodes[n][k] for k in 'zyx'], float) - fV[fidx[n]]).max() for n in inn]
        dt = sum(int(nodes[n]['t']) != int(fT[fidx[n]]) for n in inn)
        print(s, name, 'p14', len(nodes), 'in_full', len(inn), 'full', len(fids), 'dropped', len(fids) - len(inn),
              'maxdpos', round(float(max(dpos)), 3) if dpos else None, 'n_dpos>0.5', int(sum(d > 0.5 for d in dpos)), 'dt_mismatch', dt,
              'fullE', len(fE), flush=True)
