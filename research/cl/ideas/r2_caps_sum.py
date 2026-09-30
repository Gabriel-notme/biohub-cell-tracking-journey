import json, sys
from collections import Counter, defaultdict
R = json.load(open('/workspace/cl/ideas/r2_caps_out.json'))
EMB = ['44b6', '6bba']
ec = Counter(); cu = Counter()
for r in R:
    for k in r['ec']: ec[tuple(k[:-1])] += k[-1]
    for k in r['cu']: cu[tuple(k[:-1])] += k[-1]
sec = sys.argv[1] if len(sys.argv) > 1 else 'all'


def show(title, c):
    print('==', title)
    keys = sorted({k[1:] for k in c}, key=lambda k: -sum(c[(e,) + k] for e in EMB))
    for k in keys:
        print('  %-40s %6d %6d' % (' '.join(map(str, k)), c[('44b6',) + k], c[('6bba',) + k]))


if sec in ('all', 'edge'):
    c = Counter()
    for (e, kind, cat, fb, db), v in ec.items(): c[(e, kind, cat.split('_dropdet')[0].split('_nodet')[0] if kind == 'FN' and cat.startswith('miss') else cat)] += v
    show('edge TP/FN by category (44b6 6bba)', c)
    c = Counter()
    for (e, kind, cat, fb, db), v in ec.items():
        if kind == 'FN' and cat.startswith('miss'): c[(e, cat)] += v
    show('unmatched FN: dropped pre-ILP detection within 7um?', c)
    c = Counter()
    for (e, kind, cat, fb, db), v in ec.items():
        if kind == 'FN' and not cat.startswith('miss') and not cat.startswith('flip'): c[(e, cat, fb)] += v
    show('non-flip matched FN by cat x pre-ILP fe', c)
    c = Counter()
    for (e, kind, cat, fb, db), v in ec.items():
        if kind == 'FN' and not cat.startswith('miss') and not cat.startswith('flip'): c[(e, cat, db)] += v
    show('non-flip matched FN by cat x pred dist', c)
    c = Counter(); t = Counter()
    for (e, kind, cat, fb, db), v in ec.items():
        t[(e, kind if kind == 'TP' else ('FNmiss' if cat.startswith('miss') else ('FNflip' if cat.startswith('flip') else 'FNstruct')), db)] += v
    show('GT edges by GT length bin (kind, bin)', t)
if sec in ('all', 'cu'):
    c = Counter()
    for (e, cat, fb, db, lab, rm), v in cu.items(): c[(e, cat, fb, lab)] += v
    print('== pre-ILP candidates absent from P14: cat fe label')
    keys = sorted({k[1:3] for k in c})
    for k in keys:
        s = '  %-12s %-6s' % k
        for e in EMB:
            s += ' | %s TP %5d FPe %5d U %6d prec %.2f' % (e, c[(e,) + k + ('TP',)], c[(e,) + k + ('FPe',)], c[(e,) + k + ('U',)],
                                                           c[(e,) + k + ('TP',)] / max(1, c[(e,) + k + ('TP',)] + c[(e,) + k + ('FPe',)]))
        print(s)
if sec in ('all', 'div'):
    d = Counter()
    for r in R:
        for x in r['drows']: d[(x['emb'], 'TP' if x['tp'] else x['cls'])] += 1
    show('GT divisions', d)
    f = Counter()
    for r in R:
        for k in r['forks']: f[tuple(k[:-1])] += k[-1]
    show('pred forks', f)
