"""Score all graphs in a dir and dump per-movie official rows. usage: rows_dump.py <graph_dir> <out.json>"""
import os, sys, json, glob
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
os.environ['POLARS_MAX_THREADS'] = '2'
def job(p):
    import evalx
    nodes, edges = evalx.load_graph_json(p)
    return evalx.score_movie(Path(p).stem, nodes, edges)
if __name__ == '__main__':
    ps = sorted(glob.glob(sys.argv[1] + '/*.json'))
    with Pool(32) as pool: rows = pool.map(job, ps)
    json.dump(rows, open(sys.argv[2], 'w')); print('rows', len(rows))
