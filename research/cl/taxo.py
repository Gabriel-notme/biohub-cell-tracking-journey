"""Edge error taxonomy for a graph dir. usage: taxo.py <list> <graph_dir>"""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import Counter
from multiprocessing import Pool


def job(a):
    name, g = a
    import evalx
    nodes, edges = evalx.load_graph_json(Path(g) / (name + '.json'))
    r = evalx.score_movie(name, nodes, edges, detail=True)
    return r['analysis']


if __name__ == '__main__':
    names = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    with Pool(36) as pool: cs = pool.map(job, [(n, sys.argv[2]) for n in names])
    tot = Counter()
    for c in cs: tot.update(c)
    print(sys.argv[2], dict(sorted(tot.items())))
