"""rule_eval wrapper: production p15_post.relinefit (as deployed) vs the critic's probe ideas.r3_pl_relinefit (w=0.5)."""
import sys
sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/cl/ideas')
WANTS_META = True
REF = {'hold36': '/workspace/runs/b5f_hold36/working/reference_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/reference_graphs',
       'audit32': '/workspace/sync3/runs/b5f_audit32/working/reference_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/reference_graphs',
       't127b': '/workspace/sync3/runs/b5f_t127b/working/reference_graphs'}


def apply(nodes, edges, impl='prod', w=0.5, name=None, set=None, fullgeff=None, zarr=None):
    if impl == 'prod':
        import p15_post
        return p15_post.relinefit(nodes, edges, REF[set] + '/' + name + '.json', fullgeff, shape=(64, 256, 256), w=w)
    import r3_pl_relinefit
    return r3_pl_relinefit.apply(nodes, edges, name=name, set=set, fullgeff=fullgeff, zarr=zarr, w=w)
