"""Graph-level comparison: my reimplementation applied to P15 final graphs vs the claimed P17 final graphs.
usage: gcmp.py '<json list of variant kwargs>' [out.json]"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl/ideas')
from pathlib import Path
from multiprocessing import Pool
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
P15 = '/workspace/cl/ps_p15_%s/graphs'; P17 = '/workspace/cl/p16/ps_p17_%s/graphs'


def load(f):
    d = json.load(open(f)); return {int(k): v for k, v in d['nodes'].items()}, d['edges']


def sig(nodes, edges):
    N = {k: (v['t'], round(v['z'], 4), round(v['y'], 4), round(v['x'], 4)) for k, v in nodes.items()}
    E = {(int(e['source_id']), int(e['target_id'])) for e in edges}
    return N, E


def job(a):
    s, name, V = a
    import chk_reimpl as m
    n15, e15 = load(Path(P15 % s) / (name + '.json')); n17, e17 = load(Path(P17 % s) / (name + '.json'))
    N15, E15 = sig(n15, e15); N17, E17 = sig(n17, e17)
    out = {'set': s, 'movie': name,
           'p15_vs_p17': {'n15': len(N15), 'n17': len(N17), 'nodes_only15': len(set(N15) - set(N17)), 'nodes_only17': len(set(N17) - set(N15)),
                          'coord_changed': sum(1 for k in set(N15) & set(N17) if N15[k] != N17[k]),
                          'edges_only15': len(E15 - E17), 'edges_only17': len(E17 - E15)}, 'v': []}
    for kw in V:
        nn, ne, st = m.apply(n15, e15, name=name, set=s, **kw)
        N, E = sig(nn, ne)
        r = {'mine_only_nodes': sorted(set(N) - set(N17)), 'p17_only_nodes': sorted(set(N17) - set(N)),
             'coord_diff': sum(1 for k in set(N) & set(N17) if N[k] != N17[k]),
             'mine_only_edges': sorted(E - E17), 'p17_only_edges': sorted(E17 - E), 'st': st}
        r['identical'] = not (r['mine_only_nodes'] or r['p17_only_nodes'] or r['coord_diff'] or r['mine_only_edges'] or r['p17_only_edges'])
        out['v'].append(r)
    return out


if __name__ == '__main__':
    V = json.loads(sys.argv[1])
    jobs = [(s, Path(f).stem, V) for s in SETS for f in sorted(glob.glob(P15 % s + '/*.json'))]
    with Pool(int(os.environ.get('RULE_POOL', '6')), maxtasksperchild=8) as p: R = p.map(job, jobs, chunksize=1)
    if len(sys.argv) > 2: json.dump(R, open(sys.argv[2], 'w'))
    tot = {k: sum(r['p15_vs_p17'][k] for r in R) for k in R[0]['p15_vs_p17']}
    print('P15 vs P17 raw:', tot, 'movies changed', sum(1 for r in R if r['p15_vs_p17']['nodes_only15'] or r['p15_vs_p17']['edges_only15'] or r['p15_vs_p17']['nodes_only17'] or r['p15_vs_p17']['edges_only17'] or r['p15_vs_p17']['coord_changed']))
    for i, kw in enumerate(V):
        vs = [r['v'][i] for r in R]
        st = {}
        for v in vs:
            for k, x in v['st'].items(): st[k] = st.get(k, 0) + x
        print('%-90s identical %d/%d | mine-only nodes %d p17-only nodes %d coord %d | mine-only edges %d p17-only edges %d | %s' % (
            json.dumps(kw), sum(v['identical'] for v in vs), len(vs), sum(len(v['mine_only_nodes']) for v in vs), sum(len(v['p17_only_nodes']) for v in vs),
            sum(v['coord_diff'] for v in vs), sum(len(v['mine_only_edges']) for v in vs), sum(len(v['p17_only_edges']) for v in vs), st), flush=True)

