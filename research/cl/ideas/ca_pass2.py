"""Rule: p_stage4 runs relink and edge_link exactly once, in a fixed order. relink features (hist_s, fut_d, fut_curd, orphan_fut...)
are computed on the fragmented pre-edge_link graph, and relink's cut-offs (cs, cd) plus edge_link's merges create new situations
that neither model revisits. Second pass on the final P13 graph: relink -> edge_link -> prune singletons (same models)."""
import sys
sys.path.insert(0, '/workspace/p56stage')
WANTS_META = True


def apply(nodes, edges, rl_th=0.65, el_th=0.4, do_rl=True, do_el=True, gap2=True, fullgeff=None, **kw):
    import relink, edge_link, prune
    st = {}
    n, e = nodes, edges
    if do_rl:
        n, e, s1 = relink.apply(n, e, edge_link.load_full(fullgeff), '/workspace/p56stage/relink_lgb.json', th=float(rl_th))
        st['rl2'] = s1.get('rl_applied', 0)
    if do_el:
        n, e, s2 = edge_link.apply(n, e, fullgeff, '/workspace/p56stage/edge_lgb.json', th=float(el_th), allow_gap2=bool(gap2))
        st['el2'] = s2.get('el_gap1', 0) + s2.get('el_gap2', 0)
    n, e, _ = prune.prune_fragments(n, e, 2)
    return n, e, st
