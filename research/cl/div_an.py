"""Characterise division FN / FP of a set of predicted graphs against GT.
For each GT division: TP/FN; for FN, classify by what the prediction does around it; nearest pred fork on the same GT lineage (dt).
For each FP fork: nearest GT division on the same GT component (dt, distance)."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np


def job(args):
    s, f = args
    name = Path(f).stem
    if not Path('/workspace/data/train/%s.geff/nodes/props' % name).exists(): return None
    import evalx
    from tracking_cellmot.division_metrics import score_divisions, _match_full, _matched_node_attrs
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    try:
        gt, n_total = evalx.load_gt(name)
    except Exception:
        return None
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    mp = _match_full(pred, gt, evalx.SCALE, 7.)
    ma = _matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    g2p = {b: a for a, b in p2g.items()}
    na = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): (int(t), np.array([z * 1.625, y * .40625, x * .40625])) for i, t, z, y, x in zip(*[na[k].to_list() for k in [K.NODE_ID, 't', 'z', 'y', 'x']])}
    ea = gt.edge_attrs(); gch = defaultdict(list); gpar = {}
    for a, b in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gch[int(a)].append(int(b)); gpar[int(b)] = int(a)
    ch = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); ch[a].append(b); par[b] = a
    forks = {n for n in nodes if len(ch[n]) >= 2}
    fork_by_g = defaultdict(list)  # GT node -> pred forks matched to it
    for fk in forks:
        if fk in p2g: fork_by_g[p2g[fk]].append(fk)

    def chain_back(g, k):
        out = []
        while g in gpar and len(out) < k: g = gpar[g]; out.append(g)
        return out

    def chain_fwd(g, k):
        out = []; fr = [g]
        for _ in range(k):
            fr = [c for x in fr for c in gch[x]]; out += fr
        return out
    tp_div = {int(d) for d, v in res.scores.items() if v}
    rows = []
    for d, v in res.scores.items():
        d = int(d); td_ = gpos[d][0]
        kids = gch[d]
        near = []
        for g in [d] + chain_back(d, 10) + chain_fwd(d, 10):
            for fk in fork_by_g.get(g, []): near.append(nodes[fk]['t'] - td_)
        cat = 'TP' if v else None
        if not v:
            if d not in g2p and not any(g in g2p for g in chain_back(d, 1)): cat = 'parent_unmatched'
            else:
                km = [g2p.get(c) for c in kids]
                if sum(x is not None for x in km) < 2:
                    # maybe grandchildren
                    cat = 'child_unmatched'
                else:
                    pp = g2p.get(d)
                    kp = [par.get(x) for x in km]
                    if any(x is None for x in kp): cat = 'child_track_start'
                    elif pp is not None and all(x == pp for x in kp): cat = 'fork_present_but_rejected'
                    elif pp is not None and any(x == pp for x in kp): cat = 'one_child_stolen'
                    else: cat = 'both_children_elsewhere'
        # distance between the two GT children at t+1
        dd = float(np.linalg.norm(gpos[kids[0]][1] - gpos[kids[1]][1])) if len(kids) >= 2 else -1
        rows.append(dict(movie=name, set=s, kind='gt', d=d, t=td_, cat=cat, near=sorted(near, key=abs)[:3], dkids=dd))
    for fk in res.fp_forks:
        fk_ = inv[int(fk)] if int(fk) in inv else None
        if fk_ is None: continue
        g = p2g.get(fk_)
        best = None
        if g is not None:
            for x in [g] + chain_back(g, 10) + chain_fwd(g, 10):
                if len(gch[x]) >= 2:
                    dt = gpos[x][0] - nodes[fk_]['t']
                    if best is None or abs(dt) < abs(best): best = dt
        rows.append(dict(movie=name, set=s, kind='fp', t=nodes[fk_]['t'], matched=g is not None, gt_div_dt=best,
                         g_has_child=(g is not None and len(gch[g]) > 0)))
    return rows


if __name__ == '__main__':
    src = sys.argv[1]; out = sys.argv[2]
    SETS = json.loads(src)
    jobs = [(s, f) for s, d in SETS.items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(64) as p: R = [r for rs in p.map(job, jobs) if rs for r in rs]
    json.dump(R, open(out, 'w'))
    for s in sorted(set(r['set'] for r in R)):
        G = [r for r in R if r['set'] == s and r['kind'] == 'gt']; F = [r for r in R if r['set'] == s and r['kind'] == 'fp']
        tp = sum(r['cat'] == 'TP' for r in G)
        print('%s movies %d GT div %d TP %d FN %d FP %d  J %.3f' % (s, len(set(r['movie'] for r in G + F)), len(G), tp, len(G) - tp, len(F), tp / max(1, len(G) + len(F))))
        print('   FN cats', Counter(r['cat'] for r in G if r['cat'] != 'TP').most_common())
        print('   FN with pred fork on lineage within 10f: dt hist', sorted(Counter(r['near'][0] for r in G if r['cat'] != 'TP' and r['near']).items()))
        print('   FP matched %d; FP with GT div on lineage within 10f: dt hist' % sum(r['matched'] for r in F), sorted(Counter(r['gt_div_dt'] for r in F if r['gt_div_dt'] is not None).items()))
