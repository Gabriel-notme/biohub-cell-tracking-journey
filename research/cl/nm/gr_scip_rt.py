"""Runtime + outcome of a FULL global re-solve (every union candidate incl. P15 edges is a variable; P15 divisions fixed) with SCIP,
learned all-category LOEO costs (gr_models/gr_lgb_train_<other>.txt): max sum (p - theta) x - Cn*y_dropped, in/out <= 1.
Compared with the decomposed per-frame assignment and with greedy (descending p) on the same weights."""
import sys, time, glob, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '/workspace/cl/nm')
import gr_common as C, gr_feat, evalx
import numpy as np, lightgbm as lgb
from collections import defaultdict
from tracking_cellmot.metrics import summarise
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'audit32': '/workspace/sync3/runs/fullgraph_audit32', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}


def greedy(U, W, fsrc, fdst):
    used_a, used_b = set(fsrc), set(fdst); sel = []
    for (a, b), w in sorted(W.items(), key=lambda kv: -kv[1]):
        if w <= 0: break
        if a in used_a or b in used_b: continue
        used_a.add(a); used_b.add(b); sel.append((a, b))
    return sel


def scip(U, W, fsrc, fdst, Cn):
    from pyscipopt import Model, quicksum
    inp = U['inp']
    m = Model(); m.hideOutput(); m.setParam('limits/time', 120)
    x = {e: m.addVar(vtype='B', obj=w) for e, w in W.items() if e[0] not in fsrc and e[1] not in fdst}
    ins = defaultdict(list); outs = defaultdict(list)
    for (a, b), v in x.items(): outs[a].append(v); ins[b].append(v)
    for L in list(ins.values()) + list(outs.values()):
        if len(L) > 1: m.addCons(quicksum(L) <= 1)
    for n in set(ins) | set(outs):
        if inp[n]: continue
        y = m.addVar(vtype='B', obj=-Cn)
        for v in ins.get(n, []) + outs.get(n, []): m.addCons(v <= y)
    m.setMaximize(); t0 = time.time(); m.optimize(); dt = time.time() - t0
    return [e for e, v in x.items() if m.getVal(v) > 0.5], dt, len(x), m.getGap()


for s, pat in [('hold36', '6bba_2312ac41'), ('audit32', '6bba_f17befbc'), ('t127b', '6bba*')]:
    f = sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/%s.json' % (s, pat)), key=lambda p: -len(open(p).read()))[0]
    name = f.split('/')[-1][:-5]
    nodes, edges = evalx.load_graph_json(f)
    full = C.load_full(FULL[s] + '/' + name + '.geff')
    t0 = time.time(); U = C.build_union(nodes, edges, full, radius=10, knn_all=True)
    keys, X = gr_feat.features(U, nodes, edges, full, gr_feat.ilp_sets(s, name)); tf = time.time() - t0
    bst = lgb.Booster(model_file='/workspace/cl/nm/gr_models/gr_lgb_train_44b6.txt')
    p = bst.predict(X)
    divs = C.division_structure(nodes, edges); ix = U['ix']
    fsrc = {ix[a] for a in divs}; fdst = {ix[b] for bs in divs.values() for b in bs}
    div_edges = [e for e in edges if int(e['source_id']) in divs]
    base = evalx.score_movie(name, nodes, edges); sb = summarise([base])['score']
    for th in [0.5]:
        W = {k: float(q - th) for k, q in zip(keys, p)}
        Wp = {k: w for k, w in W.items() if w > 0}
        for lab, fn in [('greedy', lambda: (greedy(U, Wp, fsrc, fdst), 0, 0, 0)), ('assign', lambda: (C.solve_assign(U, Wp, fsrc, fdst), 0, 0, 0)),
                        ('scipCn0', lambda: scip(U, Wp, fsrc, fdst, 0.0)), ('scipAll', lambda: scip(U, W, fsrc, fdst, 0.05))]:
            t1 = time.time(); sel, dts, nv, gap = fn(); t2 = time.time()
            nn, ne = C.to_graph_dict(U, nodes, sel, div_edges)
            r = evalx.score_movie(name, nn, ne)
            p15e = {(ix[int(e['source_id'])], ix[int(e['target_id'])]) for e in edges}
            print(name, len(nodes), 'cands', len(keys), 'feat %.1fs' % tf, lab, 'th', th, 'solve %.2fs' % (t2 - t1), 'vars', nv, 'gap', gap,
                  'sel', len(sel), 'kept', sum(1 for e in sel if e in p15e), 'dTP', r['edge_tp'] - base['edge_tp'], 'dFP', r['edge_fp'] - base['edge_fp'],
                  'dNodes', r['num_pred_nodes'] - base['num_pred_nodes'], 'dScore %+.5f' % (summarise([r])['score'] - sb), flush=True)
