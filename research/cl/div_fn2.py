"""For FN GT divisions of type stolen/start (B5 graphs): would div_complete.candidates() generate them? If not, why?
Also q-track anatomy for stolen FNs: q start relative to t_d, whether q's pre-division nodes are GT-matched (another real cell) or unmatched (duplicate)."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
SETS = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs',
        't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs', 't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs',
        'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs'}


def job(args):
    s, f = args
    name = Path(f).stem
    if not Path('/workspace/data/train/%s.geff/nodes/props' % name).exists(): return []
    import evalx
    from tracking_cellmot.division_metrics import score_divisions, _match_full, _matched_node_attrs
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    try: gt, _ = evalx.load_gt(name)
    except Exception: return []
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    g2p = {b: a for a, b in p2g.items()}
    na = gt.node_attrs(attr_keys=[K.NODE_ID, 't'])
    gt_t = dict(zip([int(x) for x in na[K.NODE_ID].to_list()], [int(x) for x in na['t'].to_list()]))
    ea = gt.edge_attrs(); gch = defaultdict(list); gpar = {}
    for a, b in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gch[int(a)].append(int(b)); gpar[int(b)] = int(a)
    ch = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); ch[a].append(b); par[b] = a
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}

    def tlen(n, d):  # chain length following single links
        k = 1
        while len(d.get(n, [])) == 1 if isinstance(d, defaultdict) else n in d:
            n = d[n][0] if isinstance(d, defaultdict) else d[n]; k += 1
            if k > 200: break
        return k
    rows = []
    for d, v in res.scores.items():
        d = int(d)
        if v: continue
        kids = gch[d]
        if len(kids) < 2: continue
        p = g2p.get(d)
        km = [g2p.get(c) for c in kids]
        if p is None or any(x is None for x in km): continue
        # which kid is the continuation of p?
        cont = [x for x in km if par.get(x) == p]
        other = [x for x in km if par.get(x) != p]
        if len(cont) != 1 or len(other) != 1:
            rows.append(dict(set=s, movie=name, cat='odd', n_cont=len(cont))); continue
        a, b = cont[0], other[0]
        q = par.get(b)
        typ = 'start' if q is None else 'stolen'
        why = []
        if len(ch[p]) != 1: why.append('p_outdeg%d' % len(ch[p]))
        dpb = float(np.linalg.norm(pos[p] - pos[b])); dab = float(np.linalg.norm(pos[a] - pos[b]))
        if dpb > 13: why.append('d_pb>13')
        if dab > 20: why.append('d_ab>20')
        if q is not None and len(ch[q]) != 1: why.append('q_outdeg%d' % len(ch[q]))
        # q-track anatomy
        qinfo = {}
        if q is not None:
            n = q; back = 0; matched_back = 0; other_lineage = 0
            while n is not None and back < 30:
                if n in p2g:
                    matched_back += 1
                    g = p2g[n]
                    # is g on the same GT lineage as d (ancestor chain)?
                    anc = set(); x = d
                    while x in gpar: x = gpar[x]; anc.add(x)
                    if g not in anc and g != d: other_lineage += 1
                back += 1
                pn = par.get(n)
                n = pn if (pn is not None and len(ch[pn]) == 1) else None
            qinfo = dict(q_hist=back, q_matched=matched_back, q_other=other_lineage, d_qp=float(np.linalg.norm(pos[q] - pos[p])))
        rows.append(dict(set=s, movie=name, cat=typ, why=why, d_pb=dpb, d_ab=dab, **qinfo))
    return rows


if __name__ == '__main__':
    jobs = [(s, f) for s, d in SETS.items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(64) as p: R = [r for rs in p.map(job, jobs) for r in rs]
    json.dump(R, open('/workspace/cl/div_fn2.json', 'w'))
    print('FN rows', len(R), Counter(r['cat'] for r in R))
    for typ in ['start', 'stolen']:
        Q = [r for r in R if r['cat'] == typ]
        print(typ, len(Q), 'covered by candidates (no why):', sum(not r['why'] for r in Q), 'why:', Counter(w for r in Q for w in r['why']).most_common())
        print('   d_pb quantiles', np.percentile([r['d_pb'] for r in Q], [10, 50, 90]).round(1), 'd_ab', np.percentile([r['d_ab'] for r in Q], [10, 50, 90]).round(1))
    Q = [r for r in R if r['cat'] == 'stolen']
    print('stolen q_hist dist', sorted(Counter(min(r['q_hist'], 30) for r in Q).items()))
    print('stolen q pre-div nodes matched to GT (any):', Counter(r['q_matched'] > 0 for r in Q), ' matched to OTHER lineage:', Counter(r['q_other'] > 0 for r in Q))
    print('stolen d_qp quantiles', np.percentile([r['d_qp'] for r in Q], [10, 50, 90]).round(1))
