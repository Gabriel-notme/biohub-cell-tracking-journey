"""Read-only: dump local pred structure around near-miss GT divisions (pred fork within +-3 frames, 12um, not TP).
Shows pred nodes near each GT daughter at t_g..t_g+3 with their track structure (parent/children) and GT match."""
import os, sys, json
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from collections import defaultdict
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
R = json.load(open('/workspace/cl/ideas/div_lens_rows.json'))
G = [r for r in R if r['kind'] == 'gdiv' and r['tp'] == 0 and r['near']]
import evalx, zarr
from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
K = evalx.K
for r in G:
    s, name, g = r['set'], r['movie'], r['g']
    nodes, edges = evalx.load_graph_json('/workspace/cl/ps_p13_%s/graphs/%s.json' % (s, name))
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    z = zarr.open_group('/workspace/data/train/%s.geff' % name, mode='r')
    gids = np.asarray(z['nodes/ids']).tolist(); GP = np.stack([np.asarray(z['nodes/props/%s/values' % k]) for k in 'zyx'], 1) * S
    GT_ = np.asarray(z['nodes/props/t/values']).tolist(); GE = np.asarray(z['edges/ids']).tolist()
    gpos = dict(zip(gids, GP)); gt_t = dict(zip(gids, GT_)); gch = defaultdict(list); gpar = {}
    for u, v in GE: gch[u].append(v); gpar[v] = u
    ch = defaultdict(list); par = {}
    for e in edges:
        u, v = int(e['source_id']), int(e['target_id']); ch[u].append(v); par[v] = u
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}; t = {n: int(v['t']) for n, v in nodes.items()}
    byt = defaultdict(list)
    for n in nodes: byt[t[n]].append(n)
    # GT lineage nodes: grandparent, divider, and each daughter's chain for 3 frames
    role = {}
    x = g; k = 0
    while x in gpar and k < 3: x = gpar[x]; k += 1; role[x] = 'P-%d' % k
    role[g] = 'DIV'
    for i, c in enumerate(gch[g][:2]):
        x = c; k = 1
        while True:
            role[x] = 'D%d+%d' % (i, k)
            if not gch.get(x) or k >= 4: break
            x = gch[x][0]; k += 1
    print('\n=== %s %s GT div t=%d near=%s' % (s, name, gt_t[g], r['near']))
    for tt in range(gt_t[g] - 2, gt_t[g] + 5):
        gl = [gn for gn in role if gt_t[gn] == tt]
        for gn in gl:
            near = [n for n in byt[tt] if np.linalg.norm(pos[n] - gpos[gn]) < 9]
            desc = []
            for n in sorted(near, key=lambda n: np.linalg.norm(pos[n] - gpos[gn])):
                desc.append('%d(d%.1f m=%s par=%s kids=%s)' % (n, np.linalg.norm(pos[n] - gpos[gn]), role.get(p2g.get(n), 'gt%s' % p2g.get(n)) if n in p2g else '-',
                                                              par.get(n), ch.get(n, [])))
            print('  t%2d %-6s %s' % (tt, role[gn], ' | '.join(desc)))
