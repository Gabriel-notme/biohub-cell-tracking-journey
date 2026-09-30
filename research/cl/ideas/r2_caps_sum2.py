import json
from collections import Counter
R = json.load(open('/workspace/cl/ideas/r2_caps_out.json'))
cu = Counter()
for r in R:
    for k in r['cu']: cu[tuple(k[:-1])] += k[-1]
for cat in ['E->S', 'E->TF', 'E->T1', '1->S']:
    print('==', cat)
    for db in ['<=6', '6-10', '10-14', '14-18', '>18']:
        for fb in ['>=.9', '.5-.9', '.2-.5', '<.2']:
            s = ''; tot = 0
            for e in ['44b6', '6bba']:
                c = {l: sum(v for k, v in cu.items() if k[0] == e and k[1] == cat and k[2] == fb and k[3] == db and k[4] == l) for l in ['TP', 'FPe', 'U']}
                tot += sum(c.values())
                s += ' | %s TP %4d FPe %4d U %5d' % (e, c['TP'], c['FPe'], c['U'])
            if tot: print('  %-6s %-6s' % (db, fb) + s)
print('== displaced-edge labels when the added edge is TP (cat, displaced labels)')
c = Counter()
for k, v in cu.items():
    if k[4] == 'TP': c[(k[1], k[5])] += v
for k, v in sorted(c.items()): print(k, v)
rows = [x for r in R for x in r['rows']]
print('== non-flip structural FN rows: displaced labels')
c = Counter()
for x in rows:
    c[(x['emb'], x['cat'], x.get('rm_s', '-'), x.get('rm_d', '-'), 'gdiv' if x['gdiv'] else '')] += 1
for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:40]: print(k, v)
