"""check2/reimpl: division safety of p19_dup.par_dup across par_r and graph versions. For every (src, par_r): run cutdup, forkfrag,
start_trim(2.5), term_trim(no join), then par_dup(par_r) and count forks lost, deleted fork daughters / fork parents. For each movie where
a fork is lost, score the graph before and after par_dup with the official metric (division TP/FP and score delta).
usage: gx2.py <src comma> <par_r comma> out.json"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[_k] = '1'
sys.path[:0] = ['/workspace/cl/ideas', '/workspace/cl/p16/deploy', '/workspace/code', '/workspace/cl', '/workspace/official/src']
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Pool


def forks(edges):
    c = Counter(int(e['source_id']) for e in edges)
    return {k for k, v in c.items() if v >= 2}


def job(a):
    src, s, f, PR = a
    import evalx, p17_post, p19_dup, p19r
    sys.path.insert(0, '/workspace/p19ds'); import p14_post
    name = Path(f).stem
    refp = p19r.B5[s] + '/working/reference_graphs/%s.json' % name
    n, e = evalx.load_graph_json(f)
    n, e, _ = p17_post.cutdup(n, e); n, e, _ = p17_post.forkfrag(n, e, refp)
    n, e, _ = p19_dup.start_trim(n, e, p19_dup.ref_fork_daughters(refp), r=2.5, minlen=3, iters=5)
    n, e, _ = p14_post.term_trim(n, e, join=None)
    out = defaultdict(list); par = {}
    for x in e:
        aa, bb = int(x['source_id']), int(x['target_id']); out[aa].append(bb); par[bb] = aa
    F = forks(e); res = []
    for pr in PR:
        n2, e2, k, nr = p19_dup.par_dup(n, e, r=pr, minrun=3, which='shorter')
        rm = set(n) - set(n2); lost = F - forks(e2)
        r = {'src': src, 'set': s, 'movie': name, 'par_r': pr, 'par_rm': k, 'forks_lost': len(lost),
             'rm_daughter': sum(1 for u in rm if u in par and len(out[par[u]]) >= 2), 'rm_forkparent': sum(1 for u in rm if len(out.get(u, [])) >= 2)}
        if lost:
            a0 = evalx.score_movie(name, n, e); a1 = evalx.score_movie(name, n2, e2)
            r.update(div_tp=[a0['division_tp'], a1['division_tp']], div_fp=[a0['division_fp'], a1['division_fp']],
                     adj=[a0['adj_edge_jaccard'], a1['adj_edge_jaccard']], lost_ids=sorted(lost))
        res.append(r)
    return res


if __name__ == '__main__':
    import rule_eval
    srcs = sys.argv[1].split(','); PR = [float(x) for x in sys.argv[2].split(',')]
    jobs = [(src, s, f, PR) for src in srcs for s, d in rule_eval.SRC[src].items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(int(os.environ.get('RULE_POOL', '6')), maxtasksperchild=8) as p: R = [x for rs in p.map(job, jobs, chunksize=1) for x in rs]
    json.dump(R, open(sys.argv[3], 'w'))
    for src in srcs:
        for pr in PR:
            rs = [r for r in R if r['src'] == src and r['par_r'] == pr]
            ls = [r for r in rs if r['forks_lost']]
            print('%-4s par_r %.1f movies %d par_rm %d forks_lost %d rm_daughter %d rm_forkparent %d | div TP %+d FP %+d on %s' % (
                src, pr, len(rs), sum(r['par_rm'] for r in rs), sum(r['forks_lost'] for r in rs), sum(r['rm_daughter'] for r in rs),
                sum(r['rm_forkparent'] for r in rs), sum(r['div_tp'][1] - r['div_tp'][0] for r in ls), sum(r['div_fp'][1] - r['div_fp'][0] for r in ls),
                [(r['set'], r['movie'], r['forks_lost'], r['div_tp'], r['div_fp']) for r in ls][:8]), flush=True)

