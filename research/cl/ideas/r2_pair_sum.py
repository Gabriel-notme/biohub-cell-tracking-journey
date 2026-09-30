"""Summarise r2_pair_census_or.json: per topology class / embryo, oracle value of deleting a / b and of GT-free choice rules.
value unit = edge units (FN->TP = +1, FP removed ~ +0.93, TP lost = -1) ; x 7.5e-6 ~ score on 199 movies. Division TP/FP separately."""
import json, sys
from collections import defaultdict, Counter
rows = json.load(open('/workspace/cl/ideas/r2_pair_census_or.json'))
J = 0.93


def val(o):
    tp, fp, fn, dtp, dfp = o
    return tp - J * (tp + fp + fn)


def key(r):
    return '-'.join(sorted([r['ta'], r['tb']]))


def pick(r, rule):
    """return 'a' / 'b' / None: which node the GT-free rule deletes"""
    ta, tb = r['ta'], r['tb']
    if rule == 'shorter_seg':
        if r['sa'] == r['sb']: return None
        return 'a' if r['sa'] < r['sb'] else 'b'
    if rule == 'smaller_comp':
        if r['ca'] == r['cb']: return None
        return 'a' if r['ca'] < r['cb'] else 'b'
    if rule == 'start':  # delete the S node of an S-T pair
        if ta.startswith('S') and tb.startswith('T'): return 'a'
        if tb.startswith('S') and ta.startswith('T'): return 'b'
        return None
    if rule == 'oracle':
        va, vb = val(r['oa']), val(r['ob'])
        return 'a' if va >= vb else 'b'
    return None


def agg(sel, rule):
    out = {}
    for emb in ['44b6', '6bba']:
        v = 0.; n = 0; dv = Counter(); pos = neg = 0
        for r in sel:
            if not r['m'].startswith(emb): continue
            w = pick(r, rule)
            if w is None: continue
            o = r['o' + w]; x = val(o)
            if rule == 'oracle' and x <= 0: continue
            v += x; n += 1; dv['dtp'] += o[3]; dv['dfp'] += o[4]; pos += x > 1e-9; neg += x < -1e-9
        out[emb] = (n, round(v, 1), pos, neg, dict(dv))
    return out


groups = defaultdict(list)
for r in rows: groups[key(r)].append(r)
for k, sel in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    if len(sel) < 10: continue
    print('==', k, len(sel))
    for rule in ['oracle', 'shorter_seg', 'smaller_comp', 'start']:
        print('   %-12s %s' % (rule, agg(sel, rule)))
# sub-partitions a priori: run length (1, 2-3, >=4), dz==0 vs >0, same component, short seg (<=10)
for k in ['T-T', 'S-T']:
    sel = groups[k]
    for nm, f in [('run1', lambda r: r['run'] == 1), ('run2-3', lambda r: 2 <= r['run'] <= 3), ('run4+', lambda r: r['run'] >= 4),
                  ('minseg<=5', lambda r: min(r['sa'], r['sb']) <= 5), ('minseg6-15', lambda r: 6 <= min(r['sa'], r['sb']) <= 15),
                  ('minseg>15', lambda r: min(r['sa'], r['sb']) > 15), ('same_comp', lambda r: r['same']), ('d<=2', lambda r: r['d'] <= 2.0),
                  ('d2-3.5', lambda r: r['d'] > 2.0)]:
        ss = [r for r in sel if f(r)]
        print('%s %-11s n=%4d' % (k, nm, len(ss)), 'shorter:', agg(ss, 'shorter_seg'), 'oracle:', agg(ss, 'oracle'))
