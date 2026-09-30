"""P21 candidates on top of P14 graphs, using the DEPLOYED modules (/workspace/p19ds): p15_post.tbext(pmin) -> p15_post.relinefit
(w=0.5) -> optional P20 clean-up (ideas/p19r = p17 cutdup+forkfrag, p19_dup, yx border stubs). clean=False, pmin=0.8 reproduces P15;
clean=True, pmin=0.8 reproduces P20."""
import sys
sys.path.insert(0, '/workspace/cl/ideas'); sys.path.insert(0, '/workspace/cl/p16/deploy'); sys.path.insert(0, '/workspace/p19ds')
WANTS_META = True
REF = {'hold36': '/workspace/runs/b5f_hold36/working/reference_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/reference_graphs',
       'audit32': '/workspace/sync3/runs/b5f_audit32/working/reference_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/reference_graphs',
       't127b': '/workspace/sync3/runs/b5f_t127b/working/reference_graphs'}


def apply(nodes, edges, pmin=0.8, clean=True, rlf=True, name=None, set=None, fullgeff=None, zarr=None, **kw):
    import p15_post
    st = {}
    nodes, edges, s = p15_post.tbext(nodes, edges, fullgeff, K=100, minlen=5, pmin=pmin, dup=3.5, join=True); st.update(s)
    if rlf:
        nodes, edges, s = p15_post.relinefit(nodes, edges, REF[set] + '/' + name + '.json', fullgeff, shape=(64, 256, 256), w=0.5, minsep=2.0); st.update(s)
    if clean:
        import p19r
        nodes, edges, s = p19r.apply(nodes, edges, name=name, set=set); st.update({'p19r_' + k: v for k, v in s.items()})
    return nodes, edges, st
