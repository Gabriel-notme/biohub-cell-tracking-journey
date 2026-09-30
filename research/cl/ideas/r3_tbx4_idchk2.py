"""Position agreement of P14 nodes with same-id fullgraph detections, in um; B5 lineage graph as reference."""
import sys, glob, json
import numpy as np
sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/code')
from edge_link import load_full
import evalx
S = np.array([1.625, 0.40625, 0.40625])
for s, fd, bd in [('hold36', '/workspace/runs/fullgraph_hold36', '/workspace/runs/b5f_hold36/working/lineage_graphs'),
                  ('t127a', '/workspace/sync4/runs/fullgraph_t127a', '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs')]:
    for f in sorted(glob.glob('/workspace/cl/ps_p14_%s/graphs/*.json' % s))[:2]:
        name = f.split('/')[-1][:-5]
        fids, fT, fV, fE, fprob = load_full(fd + '/' + name + '.geff')
        fidx = {int(i): j for j, i in enumerate(fids.tolist())}
        for lab, path in [('p14', f), ('b5', bd + '/' + name + '.json')]:
            nodes, edges = evalx.load_graph_json(path)
            inn = [n for n in nodes if n in fidx]
            d = np.array([np.linalg.norm((np.array([nodes[n][k] for k in 'zyx'], float) - fV[fidx[n]]) * S) for n in inn])
            dtm = np.array([int(nodes[n]['t']) != int(fT[fidx[n]]) for n in inn])
            ids = np.array(sorted(nodes)); fmax = int(fids.max())
            print(s, name, lab, 'n', len(nodes), 'in', len(inn), 'd_um q50/q90/q99/max', np.round(np.quantile(d, [.5, .9, .99, 1]), 2).tolist(),
                  '>3.5um', int((d > 3.5).sum()), 'tmis', int(dtm.sum()), 'ids>fmax', int((ids > fmax).sum()), 'not_in_full', len(nodes) - len(inn),
                  'keys', sorted(nodes[inn[0]].keys()), flush=True)
