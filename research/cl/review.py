"""Rigorous comparison of exact-run submission CSVs with the official metric.
usage: review.py score <cfg> <set> <csv>          -> caches per-movie rows in /workspace/cl/rev/<cfg>_<set>.json
       review.py cmp <cfgA> <cfgB> [sets=hold36,prev4,audit32] [clean=hold36,prev4]
"""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from multiprocessing import Pool
import numpy as np
REV = Path('/workspace/cl/rev'); REV.mkdir(exist_ok=True)


def _score(args):
    ds, nodes, edges = args
    import evalx
    r = evalx.score_movie(ds, nodes, edges); r['movie'] = ds; return r


def score(cfg, st, csv):
    import pandas as pd
    df = pd.read_csv(csv)
    assert list(df.columns) == ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id'], df.columns
    assert (df['id'].values == np.arange(len(df))).all(), 'ids not consecutive'
    jobs = []
    for ds, g in df.groupby('dataset'):
        n = g[g.row_type == 'node']; e = g[g.row_type == 'edge']
        nodes = {int(r.node_id): {'t': int(r.t), 'z': float(r.z), 'y': float(r.y), 'x': float(r.x)} for r in n.itertuples()}
        edges = [{'source_id': int(s), 'target_id': int(t)} for s, t in zip(e.source_id, e.target_id)]
        jobs.append((ds, nodes, edges))
    with Pool(min(36, len(jobs))) as pool: rows = pool.map(_score, jobs)
    json.dump(rows, open(REV / ('%s_%s.json' % (cfg, st)), 'w'))
    from tracking_cellmot.metrics import summarise
    s = summarise(rows)
    print('%s %s score %.6f adjE %.6f div %d/%d/%d' % (cfg, st, s['score'], s['adj_edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn']))


def cmp(A, B, sets, clean):
    from tracking_cellmot.metrics import summarise
    ra = {s: {r['movie']: r for r in json.load(open(REV / ('%s_%s.json' % (A, s))))} for s in sets}
    rb = {s: {r['movie']: r for r in json.load(open(REV / ('%s_%s.json' % (B, s))))} for s in sets}
    print('==== %s -> %s' % (A, B))
    for s in sets:
        assert set(ra[s]) == set(rb[s]), s
        a = summarise(list(ra[s].values())); b = summarise(list(rb[s].values()))
        ea = [sum(r[k] for r in ra[s].values()) for k in ('edge_tp', 'edge_fp', 'edge_fn')]; eb = [sum(r[k] for r in rb[s].values()) for k in ('edge_tp', 'edge_fp', 'edge_fn')]
        d = [rb[s][m]['adj_edge_jaccard'] - ra[s][m]['adj_edge_jaccard'] for m in ra[s]]
        print('%-8s %s %.6f -> %.6f  d %+.6f | adjE %+.6f | edge TP/FP/FN %d/%d/%d -> %d/%d/%d | div %d/%d/%d -> %d/%d/%d | movies adjE up/down/same %d/%d/%d' % (
            s, 'clean' if s in clean else 'IN-SAMPLE', a['score'], b['score'], b['score'] - a['score'], b['adj_edge_jaccard'] - a['adj_edge_jaccard'], *ea, *eb,
            a['division_tp'], a['division_fp'], a['division_fn'], b['division_tp'], b['division_fp'], b['division_fn'],
            sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d), sum(abs(x) <= 1e-9 for x in d)))
    rng = np.random.default_rng(0)
    for name, ss in [('clean ' + '+'.join(clean), clean), ('all ' + '+'.join(sets), sets)]:
        keys = [(s, m) for s in ss for m in ra[s]]
        A_ = [ra[s][m] for s, m in keys]; B_ = [rb[s][m] for s, m in keys]
        a = summarise(A_); b = summarise(B_)
        out = []; oe = []
        for _ in range(2000):
            k = rng.integers(0, len(keys), len(keys))
            sa = summarise([A_[i] for i in k]); sb = summarise([B_[i] for i in k])
            out.append(sb['score'] - sa['score']); oe.append(sb['adj_edge_jaccard'] - sa['adj_edge_jaccard'])
        out = np.array(out); oe = np.array(oe)
        print('POOLED %-28s n=%d  %.6f -> %.6f  d %+.6f  boot CI [%+.5f, %+.5f] P>0 %.3f | edge-only d %+.6f CI [%+.5f, %+.5f] P>0 %.3f' % (
            name, len(keys), a['score'], b['score'], b['score'] - a['score'], np.quantile(out, .025), np.quantile(out, .975), (out > 0).mean(),
            b['adj_edge_jaccard'] - a['adj_edge_jaccard'], np.quantile(oe, .025), np.quantile(oe, .975), (oe > 0).mean()))
    for emb in ['44b6', '6bba']:
        keys = [(s, m) for s in sets for m in ra[s] if m.startswith(emb)]
        a = summarise([ra[s][m] for s, m in keys]); b = summarise([rb[s][m] for s, m in keys])
        print('EMBRYO %s n=%d d %+.6f (adjE %+.6f)' % (emb, len(keys), b['score'] - a['score'], b['adj_edge_jaccard'] - a['adj_edge_jaccard']))


def _scoreg(args):
    name, gdir = args
    import evalx
    nodes, edges = evalx.load_graph_json(Path(gdir) / (name + '.json'))
    r = evalx.score_movie(name, nodes, edges); r['movie'] = name; return r


def scoreg(cfg, st, gdir, lst):
    names = [l.strip() for l in open(lst) if l.strip()]
    with Pool(min(36, len(names))) as pool: rows = pool.map(_scoreg, [(n, gdir) for n in names])
    json.dump(rows, open(REV / ('%s_%s.json' % (cfg, st)), 'w'))
    from tracking_cellmot.metrics import summarise
    s = summarise(rows)
    print('%s %s score %.6f adjE %.6f div %d/%d/%d' % (cfg, st, s['score'], s['adj_edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn']))


if __name__ == '__main__':
    if sys.argv[1] == 'score':
        score(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == 'scoreg':
        scoreg(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        sets = sys.argv[4].split(',') if len(sys.argv) > 4 else ['hold36', 'prev4', 'audit32']
        clean = sys.argv[5].split(',') if len(sys.argv) > 5 else ['hold36', 'prev4']
        cmp(sys.argv[2], sys.argv[3], sets, clean)
