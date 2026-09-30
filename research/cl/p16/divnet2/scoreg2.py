"""Copy of cl/review.py scoreg writing per-movie official-metric rows to divnet2/rev/<cfg>_<set>.json."""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from multiprocessing import Pool
REV = Path('/workspace/cl/p16/divnet2/rev'); REV.mkdir(exist_ok=True)


def _scoreg(args):
    name, gdir = args
    import evalx
    nodes, edges = evalx.load_graph_json(Path(gdir) / (name + '.json'))
    r = evalx.score_movie(name, nodes, edges); r['movie'] = name; return r


if __name__ == '__main__':
    cfg, st, gdir, lst = sys.argv[1:5]
    names = [l.strip() for l in open(lst) if l.strip()]
    with Pool(min(12, len(names))) as pool: rows = pool.map(_scoreg, [(n, gdir) for n in names])
    json.dump(rows, open(REV / ('%s_%s.json' % (cfg, st)), 'w'))
    from tracking_cellmot.metrics import summarise
    s = summarise(rows)
    print('%s %s score %.6f adjE %.6f div %d/%d/%d' % (cfg, st, s['score'], s['adj_edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn']))
