"""check2/reimpl cross-graph structural check: the P19-R chain (cutdup, forkfrag, p19_dup.start_trim / term_trim / par_dup, yx border)
on other graph versions (b5, p13, p14, p15, p17). Per sub-step: nodes removed, forks (nodes with 2 children) before/after, and whether
my stricter reimplementation (par only from a true START head / true END tail) agrees with the reference.
usage: gx.py <src comma> out.json"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[_k] = '1'
sys.path[:0] = ['/workspace/cl/ideas', '/workspace/cl/p16/deploy', '/workspace/code', '/workspace/cl']
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool


def forks(edges):
    c = Counter(int(e['source_id']) for e in edges)
    return {k for k, v in c.items() if v >= 2}


def job(a):
    src, s, f = a
    import evalx, p17_post, p19_dup, p19_edge_deploy, p19r, chk2_reimpl_mine as M
    sys.path.insert(0, '/workspace/p19ds'); import p14_post
    name = Path(f).stem
    refp = p19r.B5[s] + '/working/reference_graphs/%s.json' % name
    n, e = evalx.load_graph_json(f)
    r = {'src': src, 'set': s, 'movie': name, 'n0': len(n), 'f0': len(forks(e))}
    n, e, k = p17_post.cutdup(n, e); r['cd'] = k
    n, e, k = p17_post.forkfrag(n, e, refp); r['ff'] = k
    F1 = forks(e); n1, e1 = n, e
    n, e, k = p19_dup.start_trim(n, e, p19_dup.ref_fork_daughters(refp), r=2.5, minlen=3, iters=5); r['st'] = k
    n, e, s2 = p14_post.term_trim(n, e, join=None); r['tt'] = s2['trim']
    F2 = forks(e); n2, e2 = n, e
    n, e, k, nr = p19_dup.par_dup(n, e, r=3.5, minrun=3, which='shorter'); r['par'] = k
    F3 = forks(e)
    # structural details of par_dup deletions
    out = defaultdict(list); par = {}
    for x in e2:
        aa, bb = int(x['source_id']), int(x['target_id']); out[aa].append(bb); par[bb] = aa
    rm = set(n2) - set(n)
    r['par_rm_forkparent'] = sum(1 for u in rm if len(out.get(u, [])) >= 2)
    r['par_rm_daughter'] = sum(1 for u in rm if u in par and len(out[par[u]]) >= 2)
    r['forks_lost_dup'] = len(F1 - F2); r['forks_lost_par'] = len(F2 - F3)
    r['forks_lost_par_ids'] = sorted(F2 - F3)[:10]
    # my strict variant of par on the same input
    mn, me, mk = M.parallel2(n2, e2, r=3.5)
    r['mine_par'] = mk; r['par_same'] = set(mn) == set(n)
    r['par_only_ref'] = len(set(mn) - set(n)); r['par_only_mine'] = len(set(n) - set(mn))
    rb = p19_edge_deploy.yx_border_stubs(n, e, shape_yx=(256, 256)); r['border'] = rb[2]; r['f_end'] = len(forks(rb[1]))
    return r


if __name__ == '__main__':
    import rule_eval
    srcs = sys.argv[1].split(',')
    jobs = [(src, s, f) for src in srcs for s, d in rule_eval.SRC[src].items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(int(os.environ.get('RULE_POOL', '6')), maxtasksperchild=8) as p: R = p.map(job, jobs, chunksize=1)
    json.dump(R, open(sys.argv[2], 'w'))
    for src in srcs:
        rs = [r for r in R if r['src'] == src]
        agg = {k: sum(r[k] for r in rs) for k in ['n0', 'f0', 'cd', 'ff', 'st', 'tt', 'par', 'border', 'par_rm_forkparent', 'par_rm_daughter',
                                                    'forks_lost_dup', 'forks_lost_par', 'mine_par', 'par_only_ref', 'par_only_mine', 'f_end']}
        print('%-4s movies %d %s | par identical to mine %d/%d | movies with forks lost %s' % (src, len(rs), agg, sum(r['par_same'] for r in rs), len(rs),
              [(r['set'], r['movie'], r['forks_lost_par_ids']) for r in rs if r['forks_lost_par']][:12]), flush=True)

