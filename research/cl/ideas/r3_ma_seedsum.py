import json
from collections import Counter
R = json.load(open('/workspace/cl/ideas/r3_ma_seed_rows.json'))


def cls(x):
    if x['tb'] <= 3: return 'T%d' % x['tb']
    if x['bxy'] <= 4: return 'XY'
    return 'Z'


for emb in ['44b6', '6bba']:
    rr = [(r['set'], x) for r in R if r['emb'] == emb for x in r['rows']]
    print('\n=====', emb)
    print('class side  seeds   ev  rec | near<=2: TP/FP  near<=4: TP/FP  near<=6: TP/FP | vel<=3: TP/FP  vel<=5: TP/FP | edge: TP/FP (cov) | rec w/o any cand<=6')
    for c in ['T1', 'T2', 'T3', 'XY', 'Z']:
        for side in ['s', 'e']:
            xs = [x for s, x in rr if cls(x) == c and x['side'] == side]
            if not xs: continue
            ev = [x for x in xs if x['ev']]; rec = [x for x in ev if x['rec']]
            out = '%-5s %-4s %6d %5d %4d |' % (c, side, len(xs), len(ev), len(rec))
            for r_ in [2, 4, 6]:
                tp = sum(1 for x in ev if x['d_near'] <= r_ and x['ok_near'] == 1 and x['rec'])
                fp = sum(1 for x in ev if x['d_near'] <= r_ and not (x['ok_near'] == 1 and x['rec']))
                out += '  %3d/%-3d' % (tp, fp)
            out += '    |'
            for r_ in [3, 5]:
                tp = sum(1 for x in ev if x['d_vel'] <= r_ and x['ok_vel'] == 1 and x['rec'])
                fp = sum(1 for x in ev if x['d_vel'] <= r_ and not (x['ok_vel'] == 1 and x['rec']))
                out += '  %3d/%-3d' % (tp, fp)
            tp = sum(1 for x in ev if x['has_edge'] and x['ok_edge'] == 1 and x['rec'])
            fp = sum(1 for x in ev if x['has_edge'] and not (x['ok_edge'] == 1 and x['rec']))
            out += '   | %3d/%-3d (%d)' % (tp, fp, sum(x['has_edge'] for x in xs))
            out += ' | %d' % sum(1 for x in rec if x['d_near'] > 6 and x['d_vel'] > 6)
            print(out)
    # how many seeds would each candidate class touch (node cost), for T1-3 combined
    xs = [x for s, x in rr if x['tb'] <= 3]
    print('T1-3 seeds with near<=4: %d, vel<=3: %d; step median' % (sum(x['d_near'] <= 4 for x in xs), sum(x['d_vel'] <= 3 for x in xs)))
