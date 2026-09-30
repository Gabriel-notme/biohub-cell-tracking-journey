"""Compose post-stages on top of a source graph set and score officially.
usage: pipe_eval.py <cfg_name> <src: p3|b5|dir-template-with-{set}> <stage> [<stage> ...]
stages: el:<th>[:nogap2]  relink:<th>  dfork:<K>  prune:<minlen>
"""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2'); os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import Counter
from multiprocessing import Pool
cfg, src = sys.argv[1], sys.argv[2]
STAGES = sys.argv[3:]
SETS = {'hold36': ('/workspace/hold36.txt', '/workspace/runs/fullgraph_hold36'),
        'prev4': ('/workspace/preview4.txt', '/workspace/runs/fullgraph_prev4'),
        'audit32': ('/workspace/audit32.txt', '/workspace/sync3/runs/fullgraph_audit32')}
SRC = {'p3': {'hold36': '/workspace/runs/p3_hold36/graphs', 'prev4': '/workspace/runs/p3_prev4/graphs', 'audit32': '/workspace/sync3/runs/p3_audit32/graphs'},
       'b5': {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs', 'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs'}}
ONLY = os.environ.get('ONLY_SETS')


def check(nodes, edges, orig_nodes):
    pairs = [(int(e['source_id']), int(e['target_id'])) for e in edges]
    assert len(pairs) == len(set(pairs)), 'dup'
    assert max(Counter(t for s, t in pairs).values(), default=0) <= 1, 'merge'
    assert max(Counter(s for s, t in pairs).values(), default=0) <= 2, 'outdeg'
    assert all(s in nodes and t in nodes and int(nodes[t]['t']) == int(nodes[s]['t']) + 1 for s, t in pairs), 'nonconsec'
    assert {int(v['t']) for v in nodes.values()} == {int(v['t']) for v in orig_nodes.values()}, 'frames'


def job(a):
    s, name, gdir, fdir = a
    import evalx, edge_link, relink
    nodes, edges = evalx.load_graph_json(Path(gdir) / (name + '.json'))
    orig = nodes; st = {}
    full = None
    fp = Path(fdir) / (name + '.geff')
    for stg in STAGES:
        k, *args = stg.split(':')
        if k == 'el':
            nodes, edges, x = edge_link.apply(nodes, edges, fp, os.environ.get('EL_MODEL', '/workspace/cl/edge_lgb.json'), th=float(args[0]), allow_gap2=not (len(args) > 1 and args[1] == 'nogap2'))
        elif k == 'relink':
            if full is None: full = edge_link.load_full(fp)
            nodes, edges, x = relink.apply(nodes, edges, full, os.environ.get('RL_MODEL', '/workspace/cl/relink_lgb.json'), th=float(args[0]))
        elif k == 'dfork':
            import dfork
            edges, x = dfork.resolve(nodes, edges, K=int(args[0]))
        elif k == 'prune':
            import prune
            nodes, edges, x = prune.prune_fragments(nodes, edges, int(args[0]))
        else:
            raise ValueError(stg)
        check(nodes, edges, orig)
        for kk, v in x.items():
            if isinstance(v, (int, float)): st[kk] = st.get(kk, 0) + v
    od = Path('/workspace/cl/out_%s/%s' % (cfg, s)); od.mkdir(parents=True, exist_ok=True)
    (od / (name + '.json')).write_text(json.dumps({'nodes': {str(k): v for k, v in nodes.items()}, 'edges': edges}))
    r = evalx.score_movie(name, nodes, edges); r['movie'] = name; r['stats'] = st
    return s, r


if __name__ == '__main__':
    jobs = []
    for s, (lst, f) in SETS.items():
        if ONLY and s not in ONLY.split(','): continue
        g = SRC[src][s] if src in SRC else src.format(set=s)
        for n in [l.strip() for l in open(lst) if l.strip()]: jobs.append((s, n, g, f))
    with Pool(40) as pool:
        res = pool.map(job, jobs)
    tot = Counter()
    for s in SETS:
        rows = [r for ss, r in res if ss == s]
        if not rows: continue
        for r in rows: tot.update(r['stats'])
        json.dump(rows, open('/workspace/cl/rows/%s_%s.json' % (cfg, s), 'w'))
    print(cfg, STAGES, dict(tot))
