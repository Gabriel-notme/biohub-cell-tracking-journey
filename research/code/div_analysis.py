import os, sys, json
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
import numpy as np
import evalx
from tracking_cellmot.metrics import evaluate
K = evalx.K

def div_cases(name, nodes, edges):
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: p for p, g in p2g.items()}
    ea = gt.edge_attrs()
    gedges = list(zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()))
    gsucc = defaultdict(list)
    for s, d in gedges: gsucc[int(s)].append(int(d))
    succ = defaultdict(list); par = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s
    pos = {n: np.array([nodes[n]['z'] * 1.625, nodes[n]['y'] * .40625, nodes[n]['x'] * .40625]) for n in nodes}
    out = []
    for gp, ch in gsucc.items():
        if len(ch) != 2: continue
        ga, gb = ch
        p, a, b = g2p.get(gp), g2p.get(ga), g2p.get(gb)
        rec = {'movie': name, 'p': p is not None, 'a': a is not None, 'b': b is not None}
        if p is None or a is None or b is None:
            rec['cat'] = 'missing'; out.append(rec); continue
        kids = succ.get(p, [])
        pa, pb = par.get(a), par.get(b)
        rec['dist_pa'] = float(np.linalg.norm(pos[a] - pos[p])); rec['dist_pb'] = float(np.linalg.norm(pos[b] - pos[p])); rec['dist_ab'] = float(np.linalg.norm(pos[a] - pos[b]))
        if set([a, b]) <= set(kids): rec['cat'] = 'fork_exact'
        elif len(kids) == 2: rec['cat'] = 'fork_wrong_children'
        elif a in kids or b in kids:
            other = b if a in kids else a
            po = par.get(other)
            if po is None: rec['cat'] = 'one_child_other_start'
            else:
                rec['cat'] = 'one_child_other_stolen'; rec['thief_matched'] = po in p2g; rec['thief_kids'] = len(succ.get(po, []))
                rec['thief_dist'] = float(np.linalg.norm(pos[other] - pos[po]))
                h = 0; c = po
                while c in par and h < 30: c = par[c]; h += 1
                rec['thief_hist'] = h; rec['thief_p_dist'] = float(np.linalg.norm(pos[po] - pos[p]))
                hp = 0; c = p
                while c in par and hp < 30: c = par[c]; hp += 1
                rec['p_hist'] = hp
                f = 0; c = other
                while len(succ.get(c, [])) == 1 and f < 30: c = succ[c][0]; f += 1
                rec['other_fut'] = f
        elif not kids: rec['cat'] = 'parent_ends'
        else: rec['cat'] = 'parent_links_elsewhere'
        out.append(rec)
    npred_forks = sum(1 for s, k in succ.items() if len(k) == 2)
    return out, npred_forks

if __name__ == '__main__':
    stage_dir = Path(sys.argv[1])
    allc = []; nf = 0
    for p in sorted(stage_dir.glob('*.json')):
        nodes, edges = evalx.load_graph_json(p)
        c, f = div_cases(p.stem, nodes, edges); allc += c; nf += f
    print('pred forks', nf, 'gt divisions', len(allc))
    print(Counter(r['cat'] for r in allc))
    for r in allc:
        if r['cat'] not in ('fork_exact',): print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}))
