"""Score a graph dir with the official metric. usage: score_dir.py <list.txt> <graph_dir> <out_rows.json> [nproc]"""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
from pathlib import Path
from multiprocessing import Pool
import evalx
from tracking_cellmot.metrics import summarise
lst, gdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 24
def job(name):
    nodes, edges = evalx.load_graph_json(Path(gdir) / (name + '.json'))
    r = evalx.score_movie(name, nodes, edges); r['movie'] = name; return r
if __name__ == '__main__':
    names = [l.strip() for l in open(lst) if l.strip()]
    with Pool(min(nproc, len(names))) as pool: rows = pool.map(job, names)
    json.dump(rows, open(out, 'w'))
    s = summarise(rows)
    print('%s score %.6f adjE %.6f E %.6f div %d/%d/%d eTP %d eFP %d eFN %d' % (out, s['score'], s['adj_edge_jaccard'], s['edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn'],
          sum(r['edge_tp'] for r in rows), sum(r['edge_fp'] for r in rows), sum(r['edge_fn'] for r in rows)))
