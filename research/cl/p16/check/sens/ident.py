# does chk_sens default on P15 final graphs reproduce the P17 final graphs exactly?
import sys, json, glob, os
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from multiprocessing import Pool
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
def job(a):
    s, f = a
    sys.path.insert(0, '/workspace/cl/ideas')
    import chk_sens
    name = Path(f).stem
    d = json.load(open(f)); nodes = {int(k): v for k, v in d['nodes'].items()}; edges = d['edges']
    nn, ne, st = chk_sens.apply(nodes, edges, name=name, set=s)
    g = json.load(open('/workspace/cl/p16/ps_p17_%s/graphs/%s.json' % (s, name)))
    n17 = {int(k) for k in g['nodes']}; e17 = {(int(e['source_id']), int(e['target_id'])) for e in g['edges']}
    ea = {(int(e['source_id']), int(e['target_id'])) for e in ne}
    pos_eq = all(abs(float(nn[k][c]) - float(g['nodes'][str(k)][c])) < 1e-9 for k in set(nn) & n17 for c in 'zyx')
    return name, s, set(nn) == n17, ea == e17, pos_eq, len(set(nn) ^ n17), len(ea ^ e17), st, g.get('pstage', {}).get('p17_cd'), g.get('pstage', {}).get('p17_ff'), g.get('pstage', {}).get('p17_sb'), 'p17_error' in g.get('pstage', {})
if __name__ == '__main__':
    jobs = [(s, f) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s))]
    with Pool(6) as p: R = p.map(job, jobs)
    bad = [r for r in R if not (r[2] and r[3] and r[4])]
    print('movies', len(R), 'identical', len(R) - len(bad), 'p17_errors', sum(r[11] for r in R))
    for r in bad[:20]: print('DIFF', r)
    import collections
    tot = collections.Counter()
    for r in R:
        for k, v in r[7].items():
            if isinstance(v, int): tot[k] += v
    print('posthoc totals', dict(tot), 'stage totals cd/ff/sb', sum(r[8] or 0 for r in R), sum(r[9] or 0 for r in R), sum(r[10] or 0 for r in R))
