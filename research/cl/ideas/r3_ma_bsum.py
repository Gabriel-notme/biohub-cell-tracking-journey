"""Summarise r3_ma_bound_rows.json (read-only)."""
import json, sys
from collections import Counter, defaultdict
import numpy as np
R = json.load(open('/workspace/cl/ideas/r3_ma_bound_rows.json'))
W = sum(r['edge_tp'] + r['edge_fp'] + r['edge_fn'] for r in R)
print('movies', len(R), 'W(TP+FP+FN)', W, 'edge value 1/W = %.2e' % (1 / W))
print('T values (pred max t):', Counter(r['T'] for r in R), 'gT', Counter(r['gT'] for r in R).most_common(3))


def tbin(t, T=99):
    if t <= 2: return 't%02d' % t
    if t <= 5: return 't03-05'
    if t >= T - 3: return 'T-%d' % (T - t)   # source frame for edges: T-1 is last edge
    if t >= T - 6: return 'T-6..-4'
    return 'mid'


ORD = ['t00', 't01', 't02', 't03-05', 'mid', 'T-6..-4', 'T-3', 'T-2', 'T-1', 'T-0']
for emb in ['44b6', '6bba', '']:
    rr = [r for r in R if r['emb'].startswith(emb)]
    print('\n=== GT edges by source frame, emb', emb or 'all')
    C = defaultdict(Counter); FP = Counter()
    for r in rr:
        for t, z, b, c, L in r['ge']: C[tbin(t)][c] += 1
        for t, z, b, c, nk, L in r['pe']: FP[tbin(t)] += 1
    for k in ORD:
        if k not in C: continue
        c = C[k]; n = sum(c.values()); nf = n // (1 if k.startswith('t0') and len(k) == 3 or k.startswith('T-') and len(k) == 3 else 1)
        print('%-8s n %6d  TP %.4f  src_unm %.4f dst_unm %.4f both %.4f struct %.4f | FP %5d FP/n %.4f' % (
            k, n, c['tp'] / n, c['src_unm'] / n, c['dst_unm'] / n, c['both_unm'] / n, c['struct'] / n, FP[k], FP[k] / n))
    print('--- GT nodes matched by frame')
    M = defaultdict(lambda: [0, 0])
    for r in rr:
        for t, z, b, m, d in r['gn']:
            k = tbin(t); M[k][0] += m; M[k][1] += 1
    print('  '.join('%s %.4f(n%d)' % (k, M[k][0] / M[k][1], M[k][1]) for k in ORD if k in M))

    print('--- GT nodes matched by z-slice')
    Z = defaultdict(lambda: [0, 0])
    def zb(z):
        if z <= 1: return 'z0-1'
        if z <= 4: return 'z2-4'
        if z <= 8: return 'z5-8'
        if z >= 62: return 'z62-63'
        if z >= 59: return 'z59-61'
        if z >= 55: return 'z55-58'
        return 'zmid'
    for r in rr:
        for t, z, b, m, d in r['gn']: Z[zb(z)][0] += m; Z[zb(z)][1] += 1
    print('  '.join('%s %.4f(n%d)' % (k, Z[k][0] / Z[k][1], Z[k][1]) for k in ['z0-1', 'z2-4', 'z5-8', 'zmid', 'z55-58', 'z59-61', 'z62-63'] if k in Z))
    ZE = defaultdict(Counter)
    for r in rr:
        for t, z, b, c, L in r['ge']: ZE[zb(z)][c] += 1
    print('  edges TP by z: ' + '  '.join('%s %.4f(n%d)' % (k, ZE[k]['tp'] / sum(ZE[k].values()), sum(ZE[k].values())) for k in ['z0-1', 'z2-4', 'z5-8', 'zmid', 'z55-58', 'z59-61', 'z62-63'] if k in ZE))
    print('--- GT nodes matched by xy border distance (um)')
    B = defaultdict(lambda: [0, 0]); BE = defaultdict(Counter)
    def bb(b):
        for lo, hi in [(0, 2), (2, 4), (4, 7), (7, 12)]:
            if b < hi: return 'b%d-%d' % (lo, hi)
        return 'bmid'
    for r in rr:
        for t, z, b, m, d in r['gn']: B[bb(b)][0] += m; B[bb(b)][1] += 1
        for t, z, b, c, L in r['ge']: BE[bb(b)][c] += 1
    print('  '.join('%s %.4f(n%d)' % (k, B[k][0] / B[k][1], B[k][1]) for k in ['b0-2', 'b2-4', 'b4-7', 'b7-12', 'bmid'] if k in B))
    print('  edges TP by border: ' + '  '.join('%s %.4f(n%d)' % (k, BE[k]['tp'] / sum(BE[k].values()), sum(BE[k].values())) for k in ['b0-2', 'b2-4', 'b4-7', 'b7-12', 'bmid'] if k in BE))
    print('--- unmatched GT nodes: nearest pred distance hist')
    dd = np.array([d for r in rr for t, z, b, m, d in r['gn'] if not m])
    print('  n %d  <7 %d (matched elsewhere)  7-9 %d  9-12 %d  >=12 %d' % (len(dd), (dd < 7).sum(), ((dd >= 7) & (dd < 9)).sum(), ((dd >= 9) & (dd < 12)).sum(), (dd >= 12).sum()))
    md = np.array([d for r in rr for t, z, b, m, d in r['gn'] if m])
    print('  matched distance quantiles', np.round(np.quantile(md, [.5, .9, .99, .999]), 2), 'frac >6', (md > 6).mean())
    print('--- GT divisions by t_d')
    D = defaultdict(lambda: [0, 0])
    for r in rr:
        for t, v in r.get('gdiv', []):
            k = 'early<=2' if t <= 2 else ('late>=96' if t >= 96 else 'mid')
            D[k][0] += v; D[k][1] += 1
    print('  ', dict((k, '%d/%d' % tuple(v)) for k, v in D.items()))
    fpt = Counter(); tpt = Counter()
    for r in rr:
        for t in r.get('fpf', []): fpt['early<=2' if t <= 2 else ('late>=96' if t >= 96 else 'mid')] += 1
        for t in r.get('tpf', []): tpt['early<=2' if t <= 2 else ('late>=96' if t >= 96 else 'mid')] += 1
    print('   FP forks', dict(fpt), 'TP forks', dict(tpt))
