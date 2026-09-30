"""check2/reimpl graph-level comparison on all 199 P15 graphs: my dup/border (ideas/chk2_reimpl_mine.py) vs p19r.py (p19_dup.post +
p19_edge_deploy.yx_border_stubs used as black boxes). Stage-isolated: both dup versions see the same cutdup+forkfrag graph, both border
versions see the reference dup output; 'full' = my dup then my border vs the reference chain.
usage: gcmp.py '{"start_r":..,"par_r":..,"margin":..,"bminlen":..,"variants":[{"dup":{..},"border":{..}},..]}' out.json"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_k] = '1'
sys.path[:0] = ['/workspace/cl/ideas', '/workspace/cl/p16/deploy', '/workspace/code']
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
P15 = '/workspace/cl/ps_p15_%s/graphs'


def sig(nodes, edges):
    N = {int(k): (int(v['t']), round(float(v['z']), 4), round(float(v['y']), 4), round(float(v['x']), 4)) for k, v in nodes.items()}
    E = {(int(e['source_id']), int(e['target_id'])) for e in edges}
    return N, E


def info(n, nodes, edges):
    out = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
    v = nodes[n]
    return {'id': n, 't': int(v['t']), 'zyx': [round(float(v[k]), 2) for k in 'zyx'], 'par': par.get(n), 'nch': len(out.get(n, [])),
            'flags': [k for k in v if k not in ('node_id', 't', 'z', 'y', 'x')]}


def diff(A, B, nodesA, edgesA, nodesB, edgesB, K=40):
    NA, EA = sig(nodesA, edgesA); NB, EB = sig(nodesB, edgesB)
    oa = sorted(set(NA) - set(NB)); ob = sorted(set(NB) - set(NA))
    r = {'only_' + A: [info(n, nodesA, edgesA) for n in oa[:K]], 'n_only_' + A: len(oa),
         'only_' + B: [info(n, nodesB, edgesB) for n in ob[:K]], 'n_only_' + B: len(ob),
         'coord': sum(1 for k in set(NA) & set(NB) if NA[k] != NB[k]),
         'e_only_' + A: len(EA - EB), 'e_only_' + B: len(EB - EA)}
    r['identical'] = not (oa or ob or r['coord'] or r['e_only_' + A] or r['e_only_' + B])
    return r


def job(a):
    s, f, kw = a
    import evalx, p17_post, p19_dup, p19_edge_deploy, p19r, chk2_reimpl_mine as M
    name = Path(f).stem
    refp = p19r.B5[s] + '/working/reference_graphs/%s.json' % name
    sr, pr, mg, bm = kw.get('start_r', 2.5), kw.get('par_r', 3.5), kw.get('margin', 2.0), kw.get('bminlen', 6)
    n0, e0 = evalx.load_graph_json(f)
    n1, e1, cd = p17_post.cutdup(n0, e0)
    n1, e1, ff = p17_post.forkfrag(n1, e1, refp)
    rd = p19_dup.post(n1, e1, refp, start_r=sr, par_r=pr, tt_join=None)
    rb = p19_edge_deploy.yx_border_stubs(rd[0], rd[1], shape_yx=(256, 256), minlen=bm, margin=mg)
    rp = p19r.apply(n0, e0, start_r=sr, par_r=pr, margin=mg, bminlen=bm, name=name, set=s)
    N0, _ = sig(n0, e0); N1, _ = sig(n1, e1); NR, _ = sig(rd[0], rd[1]); NB, _ = sig(rb[0], rb[1])
    res = {'set': s, 'movie': name, 'n0': len(N0), 'n1': len(N1), 'ref_dup_removed': len(N1) - len(NR), 'ref_border_removed': len(NR) - len(NB),
           'ref_dup_st': rd[2] if len(rd) > 2 and isinstance(rd[2], dict) else repr(rd[2:])[:200],
           'p19r_vs_chain': diff('p19r', 'chain', rp[0], rp[1], rb[0], rb[1]), 'v': []}
    for V in kw.get('variants', [{}]):
        md = M.dup(n1, e1, refp, start_r=sr, par_r=pr, **V.get('dup', {}))
        mb = M.border(rd[0], rd[1], minlen=bm, margin=mg, **V.get('border', {}))
        mf = M.border(md[0], md[1], minlen=bm, margin=mg, **V.get('border', {}))
        res['v'].append({'mine_dup_st': md[2], 'mine_border_removed': mb[2], 'dup': diff('ref', 'mine', rd[0], rd[1], md[0], md[1]),
                         'border': diff('ref', 'mine', rb[0], rb[1], mb[0], mb[1]), 'full': diff('ref', 'mine', rb[0], rb[1], mf[0], mf[1])})
    return res


def summ(vs):
    ks = [k for k in vs[0] if k.startswith('n_only') or k.startswith('e_only') or k == 'coord']
    return 'identical %3d/%d | %s' % (sum(v['identical'] for v in vs), len(vs), ' '.join('%s=%d' % (k, sum(v[k] for v in vs)) for k in ks))


if __name__ == '__main__':
    kw = json.loads(sys.argv[1])
    jobs = [(s, f, kw) for s in SETS for f in sorted(glob.glob(P15 % s + '/*.json'))]
    with Pool(int(os.environ.get('RULE_POOL', '6')), maxtasksperchild=8) as p: R = p.map(job, jobs, chunksize=1)
    json.dump(R, open(sys.argv[2], 'w'))
    st = defaultdict(int)
    for r in R:
        for k, x in r['ref_dup_st'].items(): st[k] += x
    print('movies', len(R), 'nodes P15', sum(r['n0'] for r in R), 'after cd+ff', sum(r['n1'] for r in R),
          'ref dup removed', sum(r['ref_dup_removed'] for r in R), dict(st), 'ref border removed', sum(r['ref_border_removed'] for r in R))
    print('p19r.apply vs explicit chain:', summ([r['p19r_vs_chain'] for r in R]))
    for i, V in enumerate(kw.get('variants', [{}])):
        st = defaultdict(int)
        for r in R:
            for k, x in r['v'][i]['mine_dup_st'].items(): st[k] += x
        print('variant', json.dumps(V), 'mine dup', dict(st), 'mine border (on ref dup graph)', sum(r['v'][i]['mine_border_removed'] for r in R))
        for key in ['dup', 'border', 'full']:
            print('   %-7s %s' % (key, summ([r['v'][i][key] for r in R])), flush=True)

