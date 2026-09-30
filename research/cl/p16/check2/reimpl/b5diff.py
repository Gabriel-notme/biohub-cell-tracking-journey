"""check2/reimpl: explain the par_dup differences (reference vs my parallel2) at par_r = 3.5 found by gx.py (B5 graphs)."""
import sys, json
sys.path[:0] = ['/workspace/cl/ideas', '/workspace/cl/p16/deploy', '/workspace/code', '/workspace/p19ds', '/workspace/cl']
from collections import defaultdict
import evalx, p17_post, p19_dup, p14_post, p19r, chk2_reimpl_mine as M, rule_eval

R = json.load(open('/workspace/cl/p16/check2/reimpl/gx.json'))
bad = [r for r in R if not r['par_same']]
print([(r['src'], r['set'], r['movie'], r['par_only_ref'], r['par_only_mine']) for r in bad])
for r in bad:
    s, name = r['set'], r['movie']
    refp = p19r.B5[s] + '/working/reference_graphs/%s.json' % name
    n, e = evalx.load_graph_json(rule_eval.SRC[r['src']][s] + '/' + name + '.json')
    n, e, _ = p17_post.cutdup(n, e); n, e, _ = p17_post.forkfrag(n, e, refp)
    n, e, _ = p19_dup.start_trim(n, e, p19_dup.ref_fork_daughters(refp), r=2.5); n, e, _ = p14_post.term_trim(n, e, join=None)
    a = p19_dup.par_dup(n, e, r=3.5); b = M.parallel2(n, e, r=3.5)
    out = defaultdict(list); par = {}
    for x in e:
        u, v = int(x['source_id']), int(x['target_id']); out[u].append(v); par[v] = u

    def seg(u):
        h = u
        while h in par and len(out[par[h]]) == 1: h = par[h]
        L = [h]
        while len(out.get(L[-1], [])) == 1: L.append(out[L[-1]][0])
        return L
    gone_ref = set(n) - set(a[0]); gone_mine = set(n) - set(b[0])
    for lab, S in [('ref-only-deleted', gone_ref - gone_mine), ('mine-only-deleted', gone_mine - gone_ref)]:
        for u in sorted(S):
            sg = seg(u)
            print(r['src'], name, lab, u, 't', n[u]['t'], '| segment head', sg[0], 't0', n[sg[0]]['t'], 'len', len(sg),
                  'head has parent', sg[0] in par, 'tail children', len(out.get(sg[-1], [])))

