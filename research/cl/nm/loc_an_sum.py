import json, sys
import numpy as np
R = json.load(open(sys.argv[1]))


def q(a, f='%.2f'):
    a = np.array([x for x in a if x is not None], float)
    if not len(a): return 'n0'
    return 'n%d med ' % len(a) + f % np.median(a) + ' q25 ' + f % np.quantile(a, .25) + ' q75 ' + f % np.quantile(a, .75)


def frac(a, cond):
    a = [x for x in a if x is not None]
    return '%.3f' % (sum(1 for x in a if cond(x)) / max(1, len(a)))


for emb in ['44b6', '6bba', '']:
    for typ in ['U', 'M57', 'M']:
        X = [r for r in R if r['typ'] == typ and r['m'].startswith(emb)]
        print('=== %s %s n=%d' % (emb or 'all', typ, len(X)))
        print('  d', q([r['d'] for r in X]), '| |dz|', q([abs(r['dz']) for r in X]), '| |dxy|', q([np.hypot(r['dy'], r['dx']) for r in X]),
              '| frac |dz|>|dxy|', frac([abs(r['dz']) - np.hypot(r['dy'], r['dx']) for r in X], lambda x: x > 0))
        print('  dz>0 (GT deeper) frac', frac([r['dz'] for r in X], lambda x: x > 0), ' pred z (um)', q([r['P'][0] for r in X]))
        print('  spike', q([r['spike'] for r in X]), '| mid->GT', q([r['mid_g'] for r in X]), 'frac mid within 7', frac([r['mid_g'] for r in X], lambda x: x <= 7),
              '| gspike', q([r['gspike'] for r in X]), '| gspeed', q([r['gspeed'] for r in X]))
        print('  gdiv<=2 frac', frac([r['gdiv'] for r in X], lambda x: x <= 2), '| raw det ->GT', q([r['raw_g'] for r in X]), 'frac raw within7',
              frac([r['raw_g'] for r in X], lambda x: x <= 7), '| raw->cur', q([r['raw_p'] for r in X]))
        print('  dropped det->GT', q([r['drop_g'] for r in X]), 'frac<=7', frac([r['drop_g'] for r in X], lambda x: x <= 7),
              '| other pred ->GT', q([r['other_d'] for r in X]), 'frac other<=7', frac([r['other_d'] for r in X], lambda x: x <= 7),
              'other matched', frac([r['other_matched'] for r in X], lambda x: x), 'g matched', frac([r['g_matched'] for r in X], lambda x: x))
        print('  other GT node near pred', q([r['g_other'] for r in X]), 'frac<=7', frac([r['g_other'] for r in X], lambda x: x <= 7))
        print('  I_p/I_g', q([r['I_p'] / max(r['I_g'], 1) for r in X]), '| I_m/I_g', q([r['I_m'] / max(r['I_g'], 1) if r['I_m'] is not None else None for r in X]),
              '| fork', frac([r['fork'] for r in X], lambda x: x), 'no par', frac([r['has_par'] for r in X], lambda x: not x), 'no child', frac([r['n_ch'] for r in X], lambda x: x == 0))
