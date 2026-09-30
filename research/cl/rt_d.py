import sys
sys.path.insert(0, '/workspace/cl/p16/redteam_p20')
from rt_common import *
rows = json.load(open('/workspace/cl/p16/redteam_p20/rt_b.json'))
for key in ('f0.875', 'f0.800', 'f0.700', 'par4.0'):
    print(key, [(r['set'], r['movie'], r['res'][key]['fork_break']) for r in rows if r['res'][key]['fork_break'].get('par_dup')])
# detail of first par_dup break at f=0.875
r = next(r for r in rows if r['res']['f0.875']['fork_break'].get('par_dup'))
s, m = r['set'], r['movie']
n15, e15, _ = load(p15p(s, m)); shp = zshape(s, m)
F15 = forks(e15)
res = steps(n15, e15, refp(s, m), shp[-2:], fscale=0.875)
pre = dict((k, (n, e)) for k, n, e in res)
nb, eb = pre['term_trim']; na, ea = pre['par_dup']
Fa = forks(ea); Fb = forks(eb)
out = defaultdict(list); par = {}
for a_, b_ in pairs(eb): out[a_].append(b_); par[b_] = a_
for p, c in F15.items():
    if Fb.get(p) == c and Fa.get(p) != c:
        print('fork parent', p, 't', n15[p]['t'], 'children', c, 'parent kept', p in na, 'children kept', [x in na for x in c])
        gone = sorted(set(nb) - set(na), key=lambda x: int(n15[x]['t']))
        near = [x for x in gone if abs(int(n15[x]['t']) - int(n15[p]['t'])) <= 6]
        print('  par_dup deleted near:', [(x, int(n15[x]['t']), 'par', par.get(x), 'out', out.get(x)) for x in near])
# lost-fork movie at real scale: P15 stats of that fork from GT view done separately; P15 per-set stage wall time vs p19r
for sname in SETS:
    a = json.load(open('/workspace/cl/ps_p15_%s/pstage_report.json' % sname)); b = json.load(open('/workspace/cl/p16/ps_p19r_%s/pstage_report.json' % sname))
    sa = sum(x['pstage'].get('seconds', 0) for x in a['records']); sb = sum(x['pstage'].get('seconds', 0) for x in b['records'])
    print('%-8s wall P15 %.0fs P19r %.0fs | sum per-movie seconds P15 %.0f P19r %.0f' % (sname, a['seconds'], b['seconds'], sa, sb))
