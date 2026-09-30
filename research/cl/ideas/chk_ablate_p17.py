"""chk_ablate: wrapper around the DEPLOYED p17_post.apply (from /workspace/p17ds) so each rule can be switched on/off."""
import sys, importlib.util
WANTS_META = True
B5 = {'hold36': '/workspace/runs/b5f_hold36', 'prev4': '/workspace/runs/b5f_prev4', 'audit32': '/workspace/sync3/runs/b5f_audit32',
      't127a': '/workspace/sync4/runs/b5f_t127a', 't127b': '/workspace/sync3/runs/b5f_t127b'}
_spec = importlib.util.spec_from_file_location('p17_post_deployed', '/workspace/p17ds/p17_post.py')
P = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P)


def apply(nodes, edges, cd=1, ff=1, sb=1, name=None, **kw):
    refp = B5[kw['set']] + '/working/reference_graphs/%s.json' % name
    return P.apply(nodes, edges, refp, cutdup_on=cd, forkfrag_on=ff, shortbranch_on=sb)
