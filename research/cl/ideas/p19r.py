"""P19-R (robust): P17's cutdup + forkfrag (B5 constants 3.2 um / 6 nodes) WITHOUT shortbranch, then the P19 small deletion rules
(p19_dup.post: START duplicate heads within start_r, term_trim without join, parallel duplicate overlap at par_r; yx border stubs:
< minlen-node fork-free components entirely within `margin` voxels of the lateral FOV border). GT-free, deletion only."""
import sys
sys.path.insert(0, '/workspace/cl/ideas'); sys.path.insert(0, '/workspace/cl/p16/deploy')
WANTS_META = True
B5 = {'hold36': '/workspace/runs/b5f_hold36', 'prev4': '/workspace/runs/b5f_prev4', 'audit32': '/workspace/sync3/runs/b5f_audit32',
      't127a': '/workspace/sync4/runs/b5f_t127a', 't127b': '/workspace/sync3/runs/b5f_t127b'}


def apply(nodes, edges, start_r=2.5, par_r=3.5, margin=2.0, bminlen=6, name=None, set=None, **kw):
    import p17_post, p19_dup, p19_edge_deploy
    refp = B5[set] + '/working/reference_graphs/%s.json' % name
    nodes, edges, a = p17_post.cutdup(nodes, edges)
    nodes, edges, b = p17_post.forkfrag(nodes, edges, refp)
    nodes, edges, s = p19_dup.post(nodes, edges, refp, start_r=start_r, par_r=par_r, tt_join=None)
    r = p19_edge_deploy.yx_border_stubs(nodes, edges, shape_yx=(256, 256), minlen=bminlen, margin=margin)
    return r[0], r[1], {'cd': a, 'ff': b}
