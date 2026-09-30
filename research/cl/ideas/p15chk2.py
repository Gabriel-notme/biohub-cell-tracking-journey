"""rule_eval wrapper: production P15 post-steps (p15_post.tbext + p15_post.relinefit, as deployed) on P14 graphs."""
import sys
sys.path.insert(0, '/workspace/p56stage')
WANTS_META = True
REF = {'hold36': '/workspace/runs/b5f_hold36/working/reference_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/reference_graphs',
       'audit32': '/workspace/sync3/runs/b5f_audit32/working/reference_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/reference_graphs',
       't127b': '/workspace/sync3/runs/b5f_t127b/working/reference_graphs'}


def apply(nodes, edges, name=None, set=None, fullgeff=None, zarr=None):
    import p15_post
    nodes, edges, s1 = p15_post.tbext(nodes, edges, fullgeff, K=100, minlen=5, pmin=0.8, dup=3.5, join=True)
    nodes, edges, s2 = p15_post.relinefit(nodes, edges, REF[set] + '/' + name + '.json', fullgeff, shape=(64, 256, 256), w=0.5)
    return nodes, edges, dict(s1, **s2)
