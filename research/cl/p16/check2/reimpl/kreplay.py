"""check2/reimpl: replay the deployed P19-R post-steps (p19ds copies of p17_post / p19_dup / p19_edge_deploy) on the graphs the KAGGLE
P15 run produced (/workspace/kout/p15/pstage_graphs + the Kaggle run's own reference_graphs), then compare with the local full rerun
(ps_p19r_prev4) at submission resolution (rounded coordinates). Baseline: Kaggle P15 vs local P15 at the same resolution."""
import os, sys, json
sys.path[:0] = ['/workspace/p19ds']
import p17_post, p19_dup, p19_edge_deploy, p14_post
import zarr
from collections import Counter

K = '/workspace/kout/p15'; LOC15 = '/workspace/cl/ps_p15_prev4/graphs'; LOC19 = '/workspace/cl/p16/check2/reimpl/ps_p19r_prev4/graphs'
print('p19_dup/p14_post files:', p19_dup.__file__, p14_post.__file__)


def load(f):
    d = json.load(open(f)); return {int(k): v for k, v in d['nodes'].items()}, d['edges']


def cs(nodes, edges, shape):
    pos = {k: (int(v['t']),) + tuple(min(max(0, int(round(float(v[a])))), lim - 1) for a, lim in zip('zyx', shape[1:])) for k, v in nodes.items()}
    N = Counter(pos.values()); E = Counter((pos[int(e['source_id'])], pos[int(e['target_id'])]) for e in edges)
    return N, E


def d(A, B):
    return {'nodes_a_only': sum((A[0] - B[0]).values()), 'nodes_b_only': sum((B[0] - A[0]).values()),
            'edges_a_only': sum((A[1] - B[1]).values()), 'edges_b_only': sum((B[1] - A[1]).values())}


for m in sorted(os.listdir(K + '/pstage_graphs')):
    m = m[:-5]
    shape = zarr.open_group('/workspace/cl/data_prev4/%s.zarr' % m, mode='r')['0'].shape
    n, e = load('%s/pstage_graphs/%s.json' % (K, m)); refp = '%s/reference_graphs/%s.json' % (K, m)
    n0 = len(n)
    n, e, a = p17_post.cutdup(n, e); n, e, b = p17_post.forkfrag(n, e, refp)
    n, e, s = p19_dup.post(n, e, refp, start_r=2.5, par_r=3.5, tt_join=None)
    r = p19_edge_deploy.yx_border_stubs(n, e, shape_yx=tuple(shape[-2:])); n, e = r[0], r[1]
    kg19 = cs(n, e, shape); kg15 = cs(*load('%s/pstage_graphs/%s.json' % (K, m)), shape)
    lc15 = cs(*load('%s/%s.json' % (LOC15, m)), shape); lc19 = cs(*load('%s/%s.json' % (LOC19, m)), shape)
    print(m, 'kaggle P15 nodes', n0, '| replay removed cd %d ff %d dup %s border %d' % (a, b, s['p19_dup'], r[2]),
          '| kgP15 vs lcP15', d(kg15, lc15), '| kg-replay-P19R vs local-rerun-P19R', d(kg19, lc19))

