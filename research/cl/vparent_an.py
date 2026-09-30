import json, glob
import numpy as np
R = []
for f in glob.glob('/workspace/cl/vp/*.json'):
    s, m = f.split('/')[-1][:-5].split('__')
    rows = json.load(open(f))
    if not rows: continue
    nneg_s = sum(1 for r in rows if r['lab'] == 'N'); w = (rows[0]['n_neg_total'] / nneg_s) if nneg_s else 1.0
    for r in rows: r['set'] = s; r['emb'] = m[:4]; r['w'] = w if r['lab'] == 'N' else 1.0; R.append(r)


def auc(pos, neg, wneg):
    pos = np.array(pos); neg = np.array(neg); wneg = np.array(wneg)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    return float(sum(((p > neg) * wneg).sum() + 0.5 * ((p == neg) * wneg).sum() for p in pos) / (len(pos) * wneg.sum()))


for grp, sets in [('b1-seen', ('t127a', 't127b', 'audit32')), ('b1-unseen', ('hold36', 'prev4'))]:
    for emb in ['44b6', '6bba', 'all']:
        Q = [r for r in R if r['set'] in sets and (emb == 'all' or r['emb'] == emb) and r['base'] is not None]
        P = [r for r in Q if r['lab'] == 'P']; D = [r for r in Q if r['lab'] == 'D']; N = [r for r in Q if r['lab'] == 'N']
        line = '%-9s %-4s P %3d D %3d N~%6.0f | AUC(P vs N) base %.3f vp %.3f' % (grp, emb, len(P), len(D), sum(r['w'] for r in N),
               auc([r['base'] for r in P], [r['base'] for r in N], [r['w'] for r in N]), auc([r['vp'] for r in P], [r['vp'] for r in N], [r['w'] for r in N]))
        for th in [0.5, 0.8, 0.9, 0.97]:
            line += ' | vp>=%.2f P %d D %d N~%.0f' % (th, sum(r['vp'] >= th for r in P), sum(r['vp'] >= th for r in D), sum(r['w'] for r in N if r['vp'] >= th))
        print(line)
