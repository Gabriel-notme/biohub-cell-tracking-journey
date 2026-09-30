"""Is the pre-ILP edge p->b (second daughter) informative for division-completion candidates (corrected labels)?"""
import os, sys, json, glob
sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}


def job(f):
    import edge_link
    s, name = Path(f).stem.split('__')
    rows = json.load(open(f)); labs = json.load(open(f.replace('/cands/', '/clab/')))
    fids, fT, fV, fE, fprob = edge_link.load_full(Path(FULL[s]) / (name + '.geff'))
    fe = {(int(a), int(b)): float(p) for (a, b), p in zip(fE.tolist(), fprob.tolist())}
    fout = Counter(int(a) for a, b in fE.tolist())
    out = []
    for r, l in zip(rows, labs):
        if l == 'U': continue
        out.append((s, r['typ'], l, fe.get((r['p'], r['b']), -1.), fe.get((r['p'], r['a']), -1.), fout.get(r['p'], 0), r['fork']))
    return out


if __name__ == '__main__':
    with Pool(24) as pool: R = [x for xs in pool.map(job, glob.glob('/workspace/cl/cands/*.json')) for x in xs]
    for typ in ['start', 'stolen']:
        for grp in [('t127a', 't127b', 'audit32'), ('hold36', 'prev4')]:
            rr = [x for x in R if x[1] == typ and x[0] in grp]
            for lab in ['P', 'D', 'N']:
                q = [x for x in rr if x[2] == lab]
                if not q: continue
                has_pb = sum(1 for x in q if x[3] >= 0); fork2 = sum(1 for x in q if x[5] >= 2)
                hi = sum(1 for x in q if x[3] >= 0 and x[6] >= 0.5)
                print('%-6s %-26s %s n=%6d  fg p->b edge %5d (%.3f)  fg outdeg2 at p %5d (%.3f)  fg p->b & b1>=.5 %d' % (typ, '+'.join(grp), lab, len(q), has_pb, has_pb / len(q), fork2, fork2 / len(q), hi))
