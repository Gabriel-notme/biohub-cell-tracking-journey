"""P19 small add-on (after P17 clean-up): p19_dup.post (START-side duplicate heads at 2.5 um + term_trim without join + parallel
duplicate overlap at 3.5 um) followed by p19_edge_deploy.yx_border_stubs (< 6-node fork-free components entirely within 2 voxels of
the lateral FOV border). Deletion only, GT-free."""
import sys
sys.path.insert(0, '/workspace/cl/ideas')
WANTS_META = True
B5 = {'hold36': '/workspace/runs/b5f_hold36', 'prev4': '/workspace/runs/b5f_prev4', 'audit32': '/workspace/sync3/runs/b5f_audit32',
      't127a': '/workspace/sync4/runs/b5f_t127a', 't127b': '/workspace/sync3/runs/b5f_t127b'}


def apply(nodes, edges, dup=1, border=1, name=None, set=None, **kw):
    st = {}
    if dup:
        import p19_dup
        nodes, edges, s = p19_dup.post(nodes, edges, B5[set] + '/working/reference_graphs/%s.json' % name, start_r=2.5, par_r=3.5, tt_join=None)
        st.update({'dup_' + k: v for k, v in s.items()} if isinstance(s, dict) else {'dup': s})
    if border:
        import p19_edge_deploy
        r = p19_edge_deploy.yx_border_stubs(nodes, edges, shape_yx=(256, 256))
        nodes, edges = r[0], r[1]
    return nodes, edges, st
