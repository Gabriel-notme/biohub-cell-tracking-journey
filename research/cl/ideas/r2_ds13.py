"""P13 -> combo14 (== P14) -> optional r2_divstruct rule. rule='none' gives the P14 reference."""
import sys
sys.path.insert(0, '/workspace/cl')
WANTS_META = True


def apply(nodes, edges, rule='none', name=None, set=None, fullgeff=None, zarr=None, **kw):
    from ideas import combo14, r2_divstruct
    nodes, edges, st = combo14.apply(nodes, edges, fullgeff=fullgeff)
    if rule != 'none':
        nodes, edges, s2 = r2_divstruct.apply(nodes, edges, rule=rule, **kw); st.update(s2)
    return nodes, edges, st
