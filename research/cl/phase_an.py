import json, glob
import numpy as np
seen = {'t127a', 't127b', 'audit32'}
G, C = [], []
for f in glob.glob('/workspace/cl/phase/*.json'):
    s, m = f.split('/')[-1][:-5].split('__')
    for r in json.load(open(f)):
        r['set'] = s; r['emb'] = m[:4]
        (G if r['kind'] == 'gt' else C).append(r)


def wauc(pos, neg, w=None):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float); w = np.ones(len(neg)) if w is None else np.asarray(w, float)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    o = np.argsort(neg); ns = neg[o]; cw = np.cumsum(w[o]); idx = np.searchsorted(ns, pos, side='left')
    return float(np.where(idx > 0, cw[np.maximum(idx - 1, 0)], 0.0).sum() / (len(pos) * w.sum()))


print('(1) GT nodes: phase head, dividing vs non-dividing')
for grp, f in [('b1-seen', lambda r: r['set'] in seen), ('b1-unseen', lambda r: r['set'] not in seen)]:
    for emb in ['44b6', '6bba']:
        Q = [r for r in G if f(r) and r['emb'] == emb]
        pos = [r['phase'] for r in Q if r['div']]; neg = [r['phase'] for r in Q if not r['div']]
        print('  %-9s %s div %3d non %5d AUC %.3f  median div %.3f non %.3f' % (grp, emb, len(pos), len(neg), wauc(pos, neg), np.median(pos) if pos else -1, np.median(neg) if neg else -1))
print('(2) candidates (P new division vs N), negatives reweighted')
for c in C: c['w'] = (c['n_neg_total'] / max(1, c['n_neg_kept'])) if c['lab'] == 'N' else 1.0
for grp, f in [('b1-seen', lambda r: r['set'] in seen), ('b1-unseen', lambda r: r['set'] not in seen)]:
    for typ in ['start', 'stolen']:
        Q = [r for r in C if f(r) and r['typ'] == typ and r['base'] is not None]
        P = [r for r in Q if r['lab'] == 'P']; N = [r for r in Q if r['lab'] == 'N']
        line = '  %-9s %-6s P %3d N~%7.0f |' % (grp, typ, len(P), sum(r['w'] for r in N))
        for key in ['base', 'fork_vp', 'ph_p', 'ph_q', 'ph_vp', 'ph_a', 'ph_b']:
            line += ' %s %.3f' % (key, wauc([r[key] for r in P], [r[key] for r in N], [r['w'] for r in N]))
        mx = lambda r: max(r['ph_p'], r['ph_q'], r['ph_vp'])
        line += ' | max(p,q,vp) %.3f | base*max %.3f' % (wauc([mx(r) for r in P], [mx(r) for r in N], [r['w'] for r in N]),
                                                     wauc([r['base'] * mx(r) for r in P], [r['base'] * mx(r) for r in N], [r['w'] for r in N]))
        print(line)
