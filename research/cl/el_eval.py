"""Apply edge_link to graphs of validation sets and score officially.
usage: el_eval.py <cfg_name> <th> <gap2 0/1> [src: p3|<dir-template with {set}>] [model]"""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2'); os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
from pathlib import Path
from collections import Counter
from multiprocessing import Pool
cfg, th, gap2 = sys.argv[1], float(sys.argv[2]), bool(int(sys.argv[3]))
SRC = sys.argv[4] if len(sys.argv) > 4 else 'p3'
MODEL = sys.argv[5] if len(sys.argv) > 5 else '/workspace/cl/edge_lgb.txt'
SETS = {'hold36': ('/workspace/hold36.txt', '/workspace/runs/p3_hold36/graphs', '/workspace/runs/fullgraph_hold36'),
        'prev4': ('/workspace/preview4.txt', '/workspace/runs/p3_prev4/graphs', '/workspace/runs/fullgraph_prev4'),
        'audit32': ('/workspace/audit32.txt', '/workspace/sync3/runs/p3_audit32/graphs', '/workspace/sync3/runs/fullgraph_audit32')}


def check(nodes, edges):
    pairs = [(int(e['source_id']), int(e['target_id'])) for e in edges]
    assert len(pairs) == len(set(pairs)), 'dup'
    assert max(Counter(t for s, t in pairs).values(), default=0) <= 1, 'merge'
    assert max(Counter(s for s, t in pairs).values(), default=0) <= 2, 'outdeg'
    assert all(s in nodes and t in nodes and int(nodes[t]['t']) == int(nodes[s]['t']) + 1 for s, t in pairs), 'nonconsec'


def job(a):
    s, name, gdir, fdir = a
    import evalx, edge_link
    nodes, edges = evalx.load_graph_json(Path(gdir) / (name + '.json'))
    n2, e2, st = edge_link.apply(nodes, edges, Path(fdir) / (name + '.geff'), MODEL, th=th, allow_gap2=gap2)
    check(n2, e2)
    od = Path('/workspace/cl/out_%s/%s' % (cfg, s)); od.mkdir(parents=True, exist_ok=True)
    (od / (name + '.json')).write_text(json.dumps({'nodes': {str(k): v for k, v in n2.items()}, 'edges': e2}))
    r = evalx.score_movie(name, n2, e2); r['movie'] = name; r['stats'] = st
    return s, r


if __name__ == '__main__':
    jobs = []
    for s, (lst, g, f) in SETS.items():
        if SRC != 'p3': g = SRC.format(set=s)
        for n in [l.strip() for l in open(lst) if l.strip()]: jobs.append((s, n, g, f))
    with Pool(40) as pool:
        res = pool.map(job, jobs)
    Path('/workspace/cl/rows').mkdir(exist_ok=True)
    tot = Counter()
    for s in SETS:
        rows = [r for ss, r in res if ss == s]
        for r in rows: tot.update({k: v for k, v in r['stats'].items()})
        json.dump(rows, open('/workspace/cl/rows/%s_%s.json' % (cfg, s), 'w'))
    print(cfg, 'th', th, 'gap2', gap2, dict(tot))
