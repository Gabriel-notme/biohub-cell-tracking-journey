"""Oracle (GT-labelled) global re-solve of P15 edges, P15 divisions fixed. Modes:
cut  : drop evaluable-negative P15 edges only
add  : keep all P15 edges, add GT-positive candidates between free ends / free starts (optimal assignment)
full : optimal per-frame assignment over all candidates with oracle weights (pos 1, P15 neutral 0.01, others 0)
drop=1 adds dropped pre-ILP detections to the node pool; radius=R adds all pairs within R um as candidates."""
import gr_common as C
WANTS_META = True


def apply(nodes, edges, mode='full', drop=0, radius=None, knn_all=0, solver='assign', name=None, set=None, fullgeff=None, zarr=None):
    full = C.load_full(fullgeff)
    U = C.build_union(nodes, edges, full, radius=radius, knn_all=bool(knn_all))
    L = C.gt_label(name, nodes, edges, U)
    C.label_edges(U, L)
    divs = C.division_structure(nodes, edges)
    ix = U['ix']
    fsrc = {ix[a] for a in divs}; fdst = {ix[b] for bs in divs.values() for b in bs}
    div_edges = [e for e in edges if int(e['source_id']) in divs]
    cand = U['cand']
    st = {'n_cand': len(cand)}
    if mode == 'cut':
        ne = [e for e in edges if int(e['source_id']) in divs or not ((ix[int(e['source_id'])], ix[int(e['target_id'])]) in cand
              and cand[(ix[int(e['source_id'])], ix[int(e['target_id'])])]['ev'] and not cand[(ix[int(e['source_id'])], ix[int(e['target_id'])])]['pos'])]
        st['removed'] = len(edges) - len(ne)
        return nodes, ne, st
    if mode == 'add':
        has_out = {ix[int(e['source_id'])] for e in edges}; has_in = {ix[int(e['target_id'])] for e in edges}
        W = {}
        for (a, b), r in cand.items():
            if r['p15'] or not r['pos']: continue
            if a in has_out or b in has_in: continue
            if not drop and (not U['inp'][a] or not U['inp'][b]): continue
            W[(a, b)] = 1.0
        sel = C.solve_assign(U, W, fsrc, fdst)
        keep = [(ix[int(e['source_id'])], ix[int(e['target_id'])]) for e in edges if int(e['source_id']) not in divs]
        nn, ne = C.to_graph_dict(U, nodes, keep + sel, div_edges)
        st['added'] = len(sel)
        return nn, ne, st
    W = {}
    for (a, b), r in cand.items():
        if not drop and (not U['inp'][a] or not U['inp'][b]): continue
        w = (1.0 if r['pos'] else 0.0) + (0.01 if r['p15'] and not (r['ev'] and not r['pos']) else 0.0)
        if w > 0: W[(a, b)] = w
    if solver == 'greedy':  # local greedy: P15 edges first, then candidates by weight, first come first served
        ua, ub = {a for a in fsrc}, {b for b in fdst}; sel = []
        for (a, b), w in sorted(W.items(), key=lambda kv: (-(cand[kv[0]]['p15']), -kv[1])):
            if a in ua or b in ub: continue
            ua.add(a); ub.add(b); sel.append((a, b))
    else:
        sel = C.solve_assign(U, W, fsrc, fdst)
    nn, ne = C.to_graph_dict(U, nodes, sel, div_edges)
    p15e = {(ix[int(e['source_id'])], ix[int(e['target_id'])]) for e in edges}
    st['kept'] = sum(1 for x in sel if x in p15e); st['new'] = len(sel) - st['kept']; st['added_nodes'] = len(nn) - len(nodes)
    return nn, ne, st
