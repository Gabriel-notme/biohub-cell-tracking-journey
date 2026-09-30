"""P15 candidates on top of P14 graphs: optional tbext (ideas/r3_tbx4 = r3_tbext: extend track ends/starts along pre-ILP edges into
dropped detections) followed by optional relinefit (p15_post). tbext kwargs are passed through as dict tb."""
import sys
sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/cl/ideas')
WANTS_META = True
REF = {'hold36': '/workspace/runs/b5f_hold36/working/reference_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/reference_graphs',
       'audit32': '/workspace/sync3/runs/b5f_audit32/working/reference_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/reference_graphs',
       't127b': '/workspace/sync3/runs/b5f_t127b/working/reference_graphs'}


def apply(nodes, edges, tb=None, rlf=True, name=None, set=None, fullgeff=None, zarr=None):
    st = {}
    if tb is not None:
        import r3_tbx4
        nodes, edges, s = r3_tbx4.apply(nodes, edges, fullgeff=fullgeff, **tb); st.update(s)
    if rlf:
        import p15_post
        nodes, edges, s = p15_post.relinefit(nodes, edges, REF[set] + '/' + name + '.json', fullgeff, shape=(64, 256, 256), w=0.5); st.update(s)
    return nodes, edges, st
