"""dv_raw: raw b1 fork probability on remaining start-type candidates of P15 (no learning): counts by threshold band, per embryo and
per set group (clean40 = hold36+prev4, never seen by b1; b1-in-sample = audit32+t127)."""
import os, sys, json, glob
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, '/workspace/cl/nm')
from dv_probe import load, greedy


def loads(f):
    return [r for r in load(f) if r['typ'] == 0]


if __name__ == '__main__':
    files = sorted(glob.glob('/workspace/cl/nm/dv_cand/*.json'))
    with Pool(24) as p: rows = [r for rs in p.map(loads, files) for r in rs]
    ex = json.load(open('/workspace/cl/nm/dv_b1_feats.json'))
    for r in rows: r.update(ex['%s|%d|%d|%d' % (r['movie'], r['p'], r['a'], r['b'])])
    cand = [r for r in rows if r['src'] == 'cand']
    grp = lambda r: 'clean40' if r['set'] in ('hold36', 'prev4') else 'b1in'
    bands = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.01]
    print('real start candidates: P (new div) / non-P by b1 band')
    for e in ['44b6', '6bba']:
        for g in ['clean40', 'b1in']:
            rr = [r for r in cand if r['emb'] == e and grp(r) == g]
            line = []
            for lo, hi in zip(bands[:-1], bands[1:]):
                x = [r for r in rr if lo <= r['b1_fork'] < hi]
                line.append('[%.1f,%.1f) %d/%d' % (lo, hi, sum(r['lab'] == 'P' for r in x), sum(r['lab'] != 'P' for r in x)))
            print(' ', e, g, ' '.join(line))
    print('synthetic births from existing forks (label tp/fp/nc) by b1 band')
    for e in ['44b6', '6bba']:
        for g in ['clean40', 'b1in']:
            rr = [r for r in rows if r['src'] == 'fork' and r['emb'] == e and grp(r) == g]
            line = []
            for lo, hi in zip(bands[:-1], bands[1:]):
                x = [r for r in rr if lo <= r['b1_fork'] < hi]
                c = Counter(r['lab'] for r in x)
                line.append('[%.1f,%.1f) tp%d fp%d nc%d' % (lo, hi, c['Ftp'], c['Ffp'], c['Fnc']))
            print(' ', e, g, ' | '.join(line))
    for e in ['44b6', '6bba']:
        rr = [r for r in cand if r['emb'] == e]
        res, hist = greedy(rr, np.array([r['b1_fork'] for r in rr]), ks=(2, 4, 8, 16, 32, 64))
        print('raw b1 greedy', e, ' '.join('%d:%d/%d' % (k, *v) for k, v in res.items()), 'thr at 16: %.3f' % hist[15][2], 'at 32: %.3f' % hist[31][2])
