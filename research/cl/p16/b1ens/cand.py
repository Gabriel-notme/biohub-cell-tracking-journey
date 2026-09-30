"""Candidate level on the ACTUAL pass-1 div_complete pool (dnlog), several tags. Labels as in divnet/analyze_dnlog.py.
usage: cand.py <tag,tag,...> [subset all|clean40] [emb filter]"""
import sys, json, glob, os
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
tags = sys.argv[1].split(','); subset = sys.argv[2] if len(sys.argv) > 2 else 'all'; ef = sys.argv[3] if len(sys.argv) > 3 else ''
cl = set(open('/workspace/hold36.txt').read().split()) | set(open('/workspace/preview4.txt').read().split())
LAB = {}
def labels(st, name):
    if (st, name) not in LAB:
        lab = {}
        for r in json.load(open('/workspace/cl/nm/dv_cand/%s__%s.json' % (st, name))):
            if r['src'] == 'cand': lab[(r['p'], r['a'], r['b'])] = r['lab'][0]
            else: lab[(r['p'], r['a'], r['b'])] = r['lab']; lab[(r['p'], r['b'], r['a'])] = r['lab']
        LAB[(st, name)] = lab
    return LAB[(st, name)]
th = {'start': 0.9, 'stolen': 0.97}; out = {}
for tag in tags:
    rows = defaultdict(list)
    for f in glob.glob('/workspace/cl/p16/b1ens/ps_%s_*/dnlog/*_1.json' % tag):
        name = os.path.basename(f).rsplit('_', 1)[0]
        if subset == 'clean40' and name not in cl: continue
        if ef and not name.startswith(ef): continue
        st = f.split('/ps_%s_' % tag)[1].split('/')[0]; lab = labels(st, name)
        for c in json.load(open(f)):
            L = lab.get((c['p'], c['a'], c['b']), '?'); y = 1 if L in ('P', 'Ftp') else (0 if L in ('N', 'X', 'Ffp') else -1)
            rows[(name[:4], c['typ'])].append((y, c['fork_b1'], c['fork_new']))
    out[tag] = {}
    for (emb, typ), R in sorted(rows.items()):
        y = np.array([r[0] for r in R]); b1 = np.array([r[1] for r in R]); nw = np.array([r[2] for r in R]); m = y >= 0
        k = max(int((b1 >= th[typ]).sum()), 1); o = np.argsort(-nw)[:k]; ob = np.argsort(-b1)[:k]
        auc = roc_auc_score(y[m], nw[m]) if 0 < y[m].sum() < m.sum() else float('nan')
        aucb = roc_auc_score(y[m], b1[m]) if 0 < y[m].sum() < m.sum() else float('nan')
        out[tag][emb + ':' + typ] = dict(n=len(R), pos=int(y[m].sum()), k=k, TP=int((y[o] == 1).sum()), FP=int((y[o] == 0).sum()), AUC=auc,
                                        b1TP=int((y[ob] == 1).sum()), b1FP=int((y[ob] == 0).sum()), b1AUC=aucb)
keys = sorted({k for t in out for k in out[t]})
print('%-8s' % 'tag' + ''.join('%-28s' % k for k in keys) + ' sumTP-FP')
for t in tags:
    line = '%-8s' % t; tot = 0
    for k in keys:
        v = out[t].get(k)
        if v: line += 'k%-3d TP %2d FP %2d AUC %.3f   ' % (v['k'], v['TP'], v['FP'], v['AUC']); tot += v['TP'] - v['FP']
        else: line += '%-28s' % '-'
    print(line + ' %+d' % tot)
t = tags[0]; line = '%-8s' % 'b1dep'
for k in keys:
    v = out[t].get(k)
    line += ('k%-3d TP %2d FP %2d AUC %.3f   ' % (v['k'], v['b1TP'], v['b1FP'], v['b1AUC'])) if v else '%-28s' % '-'
print(line + '  (deployed b1, in-sample on t127/audit32)')
json.dump(out, open('/workspace/cl/p16/b1ens/cand_%s_%s%s.json' % ('_'.join(tags)[:60], subset, ef), 'w'), indent=1)
