"""Relabel division candidates: P only if the matching GT division is NOT already recovered by the base graph;
candidates that duplicate an already-recovered GT division are labelled D (harmful)."""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
SETS = {'hold36': ('/workspace/hold36.txt', '/workspace/runs/b5f_hold36/working/lineage_graphs'),
        'prev4': ('/workspace/preview4.txt', '/workspace/runs/b5f_prev4/working/lineage_graphs'),
        'audit32': ('/workspace/audit32.txt', '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs'),
        't127a': ('/workspace/t127a.txt', '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs'),
        't127b': ('/workspace/t127b.txt', '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs')}
OUT = Path('/workspace/cl/clab')


def job(a):
    s, name, gdir = a
    import evalx
    from tracking_cellmot.metrics import evaluate
    from tracking_cellmot.division_metrics import score_divisions
    K = evalx.K
    nodes, edges = evalx.load_graph_json(Path(gdir) / (name + '.json'))
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    recovered = {int(d) for d, v in res.scores.items() if v}
    pred2, mapping2 = evalx.to_graph(nodes, edges); inv2 = {v: k for k, v in mapping2.items()}
    evaluate(pred2, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred2.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv2[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    ea = gt.edge_attrs(); gsucc = defaultdict(list); gpar = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gsucc[int(x)].append(int(y)); gpar[int(y)] = int(x)
    rows = json.load(open('/workspace/cl/cands/%s__%s.json' % (s, name)))
    labs = []
    for c in rows:
        p, a, b = c['p'], c['a'], c['b']
        gp, ga, gb = p2g.get(p), p2g.get(a), p2g.get(b)
        lab = 'U'
        if gp is not None and len(gsucc.get(gp, [])) >= 1:
            lab = 'N'
            for gd in [gp] + ([gpar[gp]] if gp in gpar else []) + list(gsucc.get(gp, [])):
                ch = gsucc.get(gd, [])
                if len(ch) != 2: continue
                lin = [set([x] + gsucc.get(x, [])) for x in ch]
                ok_a = ga is not None and any(ga in L for L in lin)
                ok_b = gb is not None and any(gb in L for L in lin)
                if ok_a and ok_b and not any(ga in L and gb in L for L in lin):
                    lab = 'D' if gd in recovered else 'P'; break
        labs.append(lab)
    OUT.mkdir(exist_ok=True)
    (OUT / ('%s__%s.json' % (s, name))).write_text(json.dumps(labs))
    return s, Counter(labs), len(res.scores), len(recovered)


if __name__ == '__main__':
    jobs = [(s, n, g) for s, (lst, g) in SETS.items() for n in [l.strip() for l in open(lst) if l.strip()]]
    tot = Counter(); gtd = Counter(); rec = Counter()
    with Pool(32) as pool:
        for s, c, ng, nr in pool.imap_unordered(job, jobs):
            for k, v in c.items(): tot[(s, k)] += v
            gtd[s] += ng; rec[s] += nr
    for k in sorted(tot): print(k, tot[k])
    print('GT divisions', dict(gtd), 'recovered by B5 base', dict(rec))
