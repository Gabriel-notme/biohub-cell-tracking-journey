"""P14 candidate post-steps on top of the final P13 graph (no GT, no learned model):
  trim = terminal-duplicate trimming (ideas/term_trim.py: r=3.5 um, END nodes only, minlen 3, iters 5, join keep_end)
  long = long-range free-end linking (ideas/pp_longlink.py: pre-ILP candidate edges end->start, edge_prob >= 0.5, > 14 um)
order: 'trim,long' | 'long,trim' | 'trim' | 'long'."""
import sys
sys.path.insert(0, '/workspace/cl/ideas')
WANTS_META = True


def apply(nodes, edges, order='trim,long', r=3.5, fe_min=0.5, min_um=14.0, name=None, set=None, fullgeff=None, zarr=None):
    import term_trim, pp_longlink
    st = {}
    for step in order.split(','):
        if step == 'trim':
            nodes, edges, s = term_trim.apply(nodes, edges, r=r, mode='end', minlen=3, iters=5, join='keep_end')
        elif step == 'long':
            nodes, edges, s = pp_longlink.apply(nodes, edges, fullgeff=fullgeff, fe_min=fe_min, min_um=min_um)
        st.update(s)
    return nodes, edges, st
