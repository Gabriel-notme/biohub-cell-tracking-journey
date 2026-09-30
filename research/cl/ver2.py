import sys, json, glob, os
from collections import defaultdict, Counter
sys.path[:0] = ['/workspace/cl', '/workspace/official/src', '/workspace/code']
import numpy as np
from multiprocessing import Pool
S = np.array([1.625, .40625, .40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
MODES = ['gapcut', 'othercut', 't0cut', 'dupcut']


def valid(nodes, edges):
    bad = Counter(); nch = Counter(); npar = Counter(); seen = set()
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id'])
        if a not in nodes or b not in nodes: bad['dangling'] += 1; continue
        if int(nodes[b]['t']) != int(nodes[a]['t']) + 1: bad['nonconsec'] += 1
        if (a, b) in seen: bad['dupedge'] += 1
        seen.add((a, b)); nch[a] += 1; npar[b] += 1
    bad['gt2children'] = sum(1 for v in nch.values() if v > 2); bad['gt1parent'] = sum(1 for v in npar.values() if v > 1)
    return bad


def job(f):
    import importlib, evalx
    m = importlib.import_module('ideas.p19v_fork')
    name = os.path.basename(f)[:-5]; s = [k for k in SETS if '_%s/' % k in f][0]
    nodes, edges = evalx.load_graph_json(f)
    res = {'movie': name, 'set': s, 'base_bad': dict(valid(nodes, edges))}
    for mode in MODES:
        nn, ne, st = m.apply(nodes, edges, mode=mode)
        gone = set(nodes) - set(nn)
        ekeep = {(int(e['source_id']), int(e['target_id'])) for e in ne}
        egone = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in ekeep]
        r = {'st': st, 'bad': dict(valid(nn, ne)), 'ngone': len(gone), 'egone': len(egone)}
        deg0 = Counter(); deg1 = Counter()
        for e in edges: deg0[int(e['source_id'])] += 1; deg0[int(e['target_id'])] += 1
        for e in ne: deg1[int(e['source_id'])] += 1; deg1[int(e['target_id'])] += 1
        r['new_isolated'] = sum(1 for n in nn if deg0[n] > 0 and deg1[n] == 0)
        if gone or egone:
            gt, _ = evalx.load_gt(name)
            ga = gt.node_attrs(attr_keys=['t', 'z', 'y', 'x'])
            gt_t = np.array(ga['t'].to_list()); gxyz = np.stack([np.array(ga[k].to_list(), float) for k in ['z', 'y', 'x']], 1) * S

            def gd(n):
                v = nodes[n]; p = np.array([float(v['z']), float(v['y']), float(v['x'])]) * S; t = int(v['t'])
                sel = gt_t == t
                return float(np.min(np.linalg.norm(gxyz[sel] - p, axis=1))) if sel.any() else 999.

            items = []
            for e in egone:
                a, b = int(e['source_id']), int(e['target_id'])
                pa = np.array([float(nodes[a][k]) for k in 'zyx']) * S; pb = np.array([float(nodes[b][k]) for k in 'zyx']) * S
                items.append({'edge': [a, b], 't': int(nodes[a]['t']), 'flags': sorted(k for k in e if k not in ('source_id', 'target_id')),
                              'src_syn': 'gap_synthetic' in nodes[a], 'dst_syn': 'gap_synthetic' in nodes[b],
                              'len_um': round(float(np.linalg.norm(pa - pb)), 2), 'gt_d_src': round(gd(a), 1), 'gt_d_dst': round(gd(b), 1)})
            r['items'] = items
            r['gone_nodes'] = [{'id': n, 't': int(nodes[n]['t']), 'syn': 'gap_synthetic' in nodes[n], 'gt_d': round(gd(n), 1)} for n in sorted(gone)]
            r['T'] = [min(int(v['t']) for v in nodes.values()), max(int(v['t']) for v in nodes.values())]
            r['gt_nodes'] = int(len(gt_t))
        res[mode] = r
    return res


if __name__ == '__main__':
    fs = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/p16/ps_p17_%s/graphs/*.json' % s))]
    with Pool(6, maxtasksperchild=8) as p: R = p.map(job, fs, chunksize=1)
    json.dump(R, open('/workspace/cl/p16/p19/v_fork/ver2.json', 'w'))
    tot = defaultdict(Counter)
    for r in R:
        for k, v in r['base_bad'].items(): tot['base'][k] += v
        for mode in MODES:
            for k, v in r[mode]['bad'].items(): tot[mode]['bad_' + k] += v
            for k, v in r[mode]['st'].items(): tot[mode]['st_' + k] += v
            for k in ('ngone', 'egone', 'new_isolated'): tot[mode][k] += r[mode][k]
            tot[mode]['movies'] += int(r[mode]['ngone'] > 0 or r[mode]['egone'] > 0)
    for k, v in tot.items(): print(k, dict(v))
