"""Critic-B probe (pipeline angle): ORDER of the two P14 steps relative to the learned link stages.
Rebuilds P14 from the frozen B5 graph on CPU (cached b1 fork probs in ideas/pp_cands, as ideas/pp_pipe.py does for P13):
div_complete -> dfork -> [tt/ll if pos=='pre_link'] -> relink -> edge_link -> post_prune -> [tt] -> [ll].
tt_pos / ll_pos: 'post' (= P14), 'pre_link' (before relink/edge_link), 'both' (tt only: before links AND after post_prune), 'off'.
Default kwargs must reproduce the given P14 graph exactly (stat diff_vs_src == 0)."""
import sys, json
from pathlib import Path
sys.path.insert(0, '/workspace/p56stage')
WANTS_META = True
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}
CD = Path('/workspace/p56stage')


def apply(nodes, edges, name=None, set=None, fullgeff=None, zarr=None, tt_pos='post', ll_pos='post'):
    import div_complete as dc, dfork, relink, edge_link, prune, p14_post
    d = json.loads((Path(B5[set]) / (name + '.json')).read_text())
    n0 = {int(k): v for k, v in d['nodes'].items()}; e0 = d['edges']
    cands = json.load(open('/workspace/cl/ideas/pp_cands/%s.json' % name))
    ne, st = dc.apply(n0, e0, cands, th_start=0.9, th_stolen=0.97, old_max=1.0)
    ne, _ = dfork.resolve(n0, ne, K=35)
    nn = n0; st = {}
    tt = lambda n_, e_: p14_post.term_trim(n_, e_, r=3.5, minlen=3, iters=5, join='keep_end')
    if tt_pos in ('pre_link', 'both'):
        nn, ne, s = tt(nn, ne); st['tt_pre'] = s['trim']
    if ll_pos == 'pre_link':
        nn, ne, s = p14_post.long_link(nn, ne, fullgeff, fe_min=0.5, min_um=14.0); st['ll_pre'] = s['ll_added']
    full = edge_link.load_full(fullgeff)
    nn, ne, _ = relink.apply(nn, ne, full, CD / 'relink_lgb.json', th=0.65)
    nn, ne, _ = edge_link.apply(nn, ne, fullgeff, CD / 'edge_lgb.json', th=0.4, allow_gap2=True)
    nn, ne, _ = prune.prune_fragments(nn, ne, 2)
    if tt_pos in ('post', 'both'):
        nn, ne, s = tt(nn, ne); st['tt_post'] = s['trim']
    if ll_pos == 'post':
        nn, ne, s = p14_post.long_link(nn, ne, fullgeff, fe_min=0.5, min_um=14.0); st['ll_post'] = s['ll_added']
    a = {(int(e['source_id']), int(e['target_id'])) for e in edges}; b = {(int(e['source_id']), int(e['target_id'])) for e in ne}
    st['diff_vs_src'] = len(a ^ b) + len(nodes.keys() ^ nn.keys())
    return nn, ne, st
