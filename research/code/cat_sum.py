import os, sys, json, glob
from pathlib import Path
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
os.environ['POLARS_MAX_THREADS'] = '2'
def job(p):
    import evalx
    nodes, edges = evalx.load_graph_json(p)
    r = evalx.score_movie(Path(p).stem, nodes, edges, detail=True)
    return r['analysis']
if __name__ == '__main__':
    ps = sorted(glob.glob(sys.argv[1] + '/*.json'))
    with Pool(32) as pool: res = pool.map(job, ps)
    c = Counter()
    for a in res: c.update(a)
    print(dict(c))
