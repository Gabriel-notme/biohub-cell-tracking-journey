"""Build P5-style graphs (P3 + relink 0.65 + edge_link 0.4) for training sets t127a/t127b/audit32."""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2'); os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from multiprocessing import Pool
SETS = {'t127a': ('/workspace/t127a.txt', '/workspace/sync4/runs/fin_p3_t127a/graphs', '/workspace/sync4/runs/fullgraph_t127a'),
        't127b': ('/workspace/t127b.txt', '/workspace/sync3/runs/fin_p3_t127b/graphs', '/workspace/sync3/runs/fullgraph_t127b'),
        'audit32': ('/workspace/audit32.txt', '/workspace/cl/ps_p3_audit32/graphs', '/workspace/sync3/runs/fullgraph_audit32')}


def job(a):
    s, name, g, f = a
    import evalx, edge_link, relink
    nodes, edges = evalx.load_graph_json(Path(g) / (name + '.json'))
    fp = Path(f) / (name + '.geff')
    nodes, edges, _ = relink.apply(nodes, edges, edge_link.load_full(fp), '/workspace/cl/relink_lgb.json', th=0.65)
    nodes, edges, _ = edge_link.apply(nodes, edges, fp, '/workspace/cl/edge_lgb.json', th=0.4)
    od = Path('/workspace/cl/p5tr/%s' % s); od.mkdir(parents=True, exist_ok=True)
    (od / (name + '.json')).write_text(json.dumps({'nodes': {str(k): v for k, v in nodes.items()}, 'edges': edges}))
    return 1


if __name__ == '__main__':
    jobs = [(s, n, g, f) for s, (lst, g, f) in SETS.items() for n in [l.strip() for l in open(lst) if l.strip()]]
    with Pool(40) as pool: print('done', sum(pool.map(job, jobs)))
