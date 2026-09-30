"""Pre-registered decision checks M1-M5 (NOTES.md) for a v2 variant run (tag) vs v1 (dnA) and the v1 A/A rerun (v1rr).
usage: judge.py <tag> [nvariants=3]"""
import sys, json, glob, os, subprocess
from collections import defaultdict
import numpy as np
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
import warnings; warnings.filterwarnings('ignore')
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
tag = sys.argv[1]; nv = int(sys.argv[2]) if len(sys.argv) > 2 else 3
H = '/workspace/cl/p16/divnet2'


def load(fmt):
    out = {}
    for s in SETS:
        for r in json.load(open(fmt % s)): out[r['movie']] = dict(r, set=s)
    return out


V1 = load('/workspace/cl/rev/dn_dnA_%s.json')
RR = load(H + '/rev/dn2_v1rr_%s.json') if os.path.exists(H + '/rev/dn2_v1rr_t127b.json') else None
C = load(H + '/rev/dn2_' + tag + '_%s.json')
ms = sorted(V1); assert set(ms) == set(C), 'movie mismatch'
rng = np.random.default_rng(12345)
BOOT = rng.integers(0, len(ms), (2000, len(ms)))


def d(base, cur, sel):
    return summarise([cur[m] for m in sel])['score'] - summarise([base[m] for m in sel])['score']


def boot(base, cur, sel):
    sel = list(sel); idx = {m: i for i, m in enumerate(ms)}
    B = rng.integers(0, len(sel), (2000, len(sel))) if len(sel) != len(ms) else BOOT
    return np.array([d(base, cur, [sel[i] for i in b]) for b in B])


def cnt(R, sel, k): return sum(R[m][k] for m in sel)


def report(base, cur, lab):
    a = 0.05 / nv
    out = {}
    dall = d(base, cur, ms); b = boot(base, cur, ms)
    lo, hi = np.quantile(b, a / 2), np.quantile(b, 1 - a / 2)
    out['M1'] = dall > 0 and lo > 0
    print('[%s] all-199 d %+.5f  Bonferroni %.2f%% CI [%+.5f, %+.5f]  (95%% CI [%+.5f, %+.5f])  P>0 %.3f  div TP %+d FP %+d' % (
        lab, dall, 100 * (1 - a), lo, hi, np.quantile(b, .025), np.quantile(b, .975), (b > 0).mean(),
        cnt(cur, ms, 'division_tp') - cnt(base, ms, 'division_tp'), cnt(cur, ms, 'division_fp') - cnt(base, ms, 'division_fp')))
    ok2 = True
    for e in ['44b6', '6bba']:
        sel = [m for m in ms if m.startswith(e)]; de = d(base, cur, sel); be = boot(base, cur, sel); ok2 &= de >= 0
        print('   embryo %s n=%d d %+.5f P>0 %.3f div TP %+d FP %+d' % (e, len(sel), de, (be > 0).mean(), cnt(cur, sel, 'division_tp') - cnt(base, sel, 'division_tp'),
                                                                 cnt(cur, sel, 'division_fp') - cnt(base, sel, 'division_fp')))
    out['M2'] = ok2
    for s in SETS:
        sel = [m for m in ms if V1[m]['set'] == s]
        print('   set %-7s n=%3d d %+.5f div TP %+d FP %+d' % (s, len(sel), d(base, cur, sel), cnt(cur, sel, 'division_tp') - cnt(base, sel, 'division_tp'),
                                                            cnt(cur, sel, 'division_fp') - cnt(base, sel, 'division_fp')))
    return out, dall


print('=' * 20, tag, 'vs v1 (dnA)')
r1, dv = report(V1, C, 'v2 - v1')
rows = []
for m in ms: rows.append(dict(V1[m], vi=0)); rows.append(dict(C[m], vi=1))
fr = H + '/rows_%s_vs_v1.json' % tag; json.dump(rows, open(fr, 'w'))
st = subprocess.run(['python3', '/workspace/cl/strict.py', fr, '1', tag + ' vs v1'], capture_output=True, text=True).stdout
print(st.rstrip()); r1['M3'] = ' PASS ' in st.split('\n')[0] + ' '
if RR is not None:
    print('=' * 20, 'A/A: v1rr vs v1'); daa = d(V1, RR, ms)
    print('   A/A all-199 d %+.5f; movies changed %d' % (daa, sum(1 for m in ms if any(RR[m][k] != V1[m][k] for k in ('edge_tp', 'edge_fp', 'division_tp', 'division_fp')))))
    print('=' * 20, tag, 'vs v1rr')
    r4, dv2 = report(RR, C, 'v2 - v1rr')
    r1['M4'] = r4['M1'] and abs(daa) < dv
else:
    r1['M4'] = None
# M5 candidate level on the real first-pass pool
lab = {}
def labels(st, name):
    L = {}
    for r in json.load(open('/workspace/cl/nm/dv_cand/%s__%s.json' % (st, name))):
        if r['src'] == 'cand': L[(r['p'], r['a'], r['b'])] = r['lab'][0]
        else: L[(r['p'], r['a'], r['b'])] = r['lab']; L[(r['p'], r['b'], r['a'])] = r['lab']
    return L
pool = {'v1': defaultdict(list), 'v2': defaultdict(list)}
for s in SETS:
    for f in glob.glob('/workspace/cl/p16/divnet/ps_dnA_%s/dnlog/*_1.json' % s):
        name = os.path.basename(f).rsplit('_', 1)[0]; L = labels(s, name)
        for key, ff in (('v1', f), ('v2', '%s/ps_%s_%s/dnlog/%s_1.json' % (H, tag, s, name))):
            for c in json.load(open(ff)):
                l = L.get((c['p'], c['a'], c['b']), '?'); y = 1 if l in ('P', 'Ftp') else (0 if l in ('N', 'X', 'Ffp') else -1)
                pool[key][(name[:4], c['typ'])].append((y, c['fork_b1'], c['fork_new']))
th = {'start': 0.9, 'stolen': 0.97}; ok5 = True; net = defaultdict(lambda: [0, 0])
print('=' * 20, 'M5 candidate level (first-pass dc pool, top-k = P15 acceptance count)')
for key in sorted(pool['v1']):
    line = '   %s %-6s' % key
    for v in ('v1', 'v2'):
        R = np.array(pool[v][key]); k = int((R[:, 1] >= th[key[1]]).sum()); o = np.argsort(-R[:, 2])[:k]
        tp = int((R[o, 0] == 1).sum()); fp = int((R[o, 0] == 0).sum()); net[(key[0], v)][0] += tp; net[(key[0], v)][1] += fp
        line += ' | %s top%d TP %d FP %d' % (v, k, tp, fp)
    print(line)
for e in ['44b6', '6bba']:
    n1 = net[(e, 'v1')][0] - .4 * net[(e, 'v1')][1]; n2 = net[(e, 'v2')][0] - .4 * net[(e, 'v2')][1]; ok5 &= n2 >= n1
    print('   %s net TP-0.4FP v1 %.1f v2 %.1f' % (e, n1, n2))
r1['M5'] = ok5
print('DECISION', tag, ' '.join('%s=%s' % kv for kv in r1.items()), '->', 'PASS' if all(v is True for v in r1.values()) else 'FAIL')
