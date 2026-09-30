"""Post-hoc isolated-component pruning with a larger minimum size on P13 outputs (equivalent to post_prune=k since it is the last step).
Per embryo and clean/in-sample split, all 199 movies."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p12ds')
from pathlib import Path
from multiprocessing import Pool


def job(a):
    s, f, k = a
    import evalx, prune
    nodes, edges = evalx.load_graph_json(f)
    if k > 2: nodes, edges, _ = prune.prune_fragments(nodes, edges, k)
    r = evalx.score_movie(Path(f).stem, nodes, edges); r['set'] = s; r['k'] = k; return r


if __name__ == '__main__':
    fs = [(s, f) for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b'] for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    KS = [2, 3, 4, 6]
    with Pool(96) as p: R = p.map(job, [(s, f, k) for k in KS for s, f in fs])
    from tracking_cellmot.metrics import summarise
    for nm, flt in [('44b6 clean', lambda r: r['movie'].startswith('44b6') and r['set'] in ('hold36', 'prev4')), ('6bba clean', lambda r: r['movie'].startswith('6bba') and r['set'] in ('hold36', 'prev4')),
                    ('44b6 all', lambda r: r['movie'].startswith('44b6')), ('6bba all', lambda r: r['movie'].startswith('6bba')), ('all199', lambda r: True)]:
        base = summarise([r for r in R if r['k'] == 2 and flt(r)])['score']
        print('%-11s base %.6f ' % (nm, base) + ' '.join('k%d %+.5f' % (k, summarise([r for r in R if r['k'] == k and flt(r)])['score'] - base) for k in KS[1:]))
