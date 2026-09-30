"""chk_sens (read-only checker, key sens): P17 post rules with overridable constants, applied to a final graph.
kwargs: rad (cutdup um), minlen (forkfrag comp size), maxk (shortbranch; branch <= maxk+1 nodes), cd/ff/sb on-off.
Graph validity checked as in p_stage11 (invalid -> input graph returned, as the stage's try/except would)."""
import sys
from collections import Counter
sys.path.insert(0, '/workspace/cl/p16/deploy')
import p17_post
WANTS_META = True
B5 = {'hold36': '/workspace/runs/b5f_hold36', 'prev4': '/workspace/runs/b5f_prev4', 'audit32': '/workspace/sync3/runs/b5f_audit32',
      't127a': '/workspace/sync4/runs/b5f_t127a', 't127b': '/workspace/sync3/runs/b5f_t127b'}


def check_graph(nodes, edges, orig_nodes):
    pairs = [(int(e['source_id']), int(e['target_id'])) for e in edges]
    assert len(pairs) == len(set(pairs)), 'dup edges'
    assert max(Counter(t for s, t in pairs).values(), default=0) <= 1, 'merge'
    assert max(Counter(s for s, t in pairs).values(), default=0) <= 2, 'outdeg'
    assert all(s in nodes and t in nodes and int(nodes[t]['t']) == int(nodes[s]['t']) + 1 for s, t in pairs), 'dangling/nonconsecutive'
    assert {int(v['t']) for v in nodes.values()} == {int(v['t']) for v in orig_nodes.values()}, 'frames'


def apply(nodes, edges, rad=3.2, minlen=6, maxk=2, cd=1, ff=1, sb=1, name=None, **meta):
    refp = B5[meta['set']] + '/working/reference_graphs/%s.json' % name
    n0, e0 = nodes, edges; st = {}
    if cd:
        nodes, edges, k = p17_post.cutdup(nodes, edges, rad=rad); st['s_cd'] = k
    if ff:
        nodes, edges, k = p17_post.forkfrag(nodes, edges, refp, minlen=minlen); st['s_ff'] = k
    if sb:
        nodes, edges, f, k = p17_post.short_branch(nodes, edges, maxk=maxk); st['s_sbf'] = f; st['s_sb'] = k
    try:
        check_graph(nodes, edges, n0)
    except AssertionError as e:
        return n0, e0, {'s_invalid': str(e)}
    return nodes, edges, st
