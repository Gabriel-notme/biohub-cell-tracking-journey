"""Candidate-level comparison on the ACTUAL first-pass div_complete pool logged by dn_patch (fork_b1 = deployed b1, fork_new = LOEO scorer).
Labels are joined from nm/dv_cand (official matching on the P15 graphs): cand rows P/D/N/X (U sampled 1%) and P15 forks Ftp/Ffp/Fnc.
Positive = P or Ftp; negative = N, X or Ffp; D/Fnc/U/unknown excluded.
usage: analyze_dnlog.py <tag> [pass=1] [subset all|clean40]"""
import sys, json, glob, os
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
tag = sys.argv[1]; ps = sys.argv[2] if len(sys.argv) > 2 else '1'; subset = sys.argv[3] if len(sys.argv) > 3 else 'all'
cl = set(open('/workspace/hold36.txt').read().split()) | set(open('/workspace/preview4.txt').read().split())
rows = defaultdict(list)
for f in glob.glob('/workspace/cl/p16/b1ens/ps_%s_*/dnlog/*_%s.json' % (tag, ps)):
    name = os.path.basename(f).rsplit('_', 1)[0]
    if subset == 'clean40' and name not in cl: continue
    st = f.split('/ps_%s_' % tag)[1].split('/')[0]
    lab = {}
    for r in json.load(open('/workspace/cl/nm/dv_cand/%s__%s.json' % (st, name))):
        if r['src'] == 'cand': lab[(r['p'], r['a'], r['b'])] = r['lab'][0]
        else: lab[(r['p'], r['a'], r['b'])] = r['lab']; lab[(r['p'], r['b'], r['a'])] = r['lab']
    for c in json.load(open(f)):
        L = lab.get((c['p'], c['a'], c['b']), '?')
        y = 1 if L in ('P', 'Ftp') else (0 if L in ('N', 'X', 'Ffp') else -1)
        rows[(name[:4], c['typ'])].append((y, c['fork_b1'], c['fork_new'], L, name))
th = {'start': 0.9, 'stolen': 0.97}
for (emb, typ), R in sorted(rows.items()):
    y = np.array([r[0] for r in R]); b1 = np.array([r[1] for r in R]); nw = np.array([r[2] for r in R]); m = y >= 0
    k = int((b1 >= th[typ]).sum())
    line = '%s %-6s n=%6d lab=%5d pos=%3d | b1 accepts %3d' % (emb, typ, len(R), m.sum(), y[m].sum(), k)
    if 0 < y[m].sum() < m.sum():
        line += ' | AUC b1dep %.3f new %.3f' % (roc_auc_score(y[m], b1[m]), roc_auc_score(y[m], nw[m]))
    for nm, s in [('b1dep', b1), ('new', nw)]:
        # global top-k over the embryo (k = b1 acceptance count), count labelled TP/FP
        o = np.argsort(-s)[:max(k, 1)]
        line += ' | %s top%d TP %d FP %d' % (nm, k, int((y[o] == 1).sum()), int((y[o] == 0).sum()))
    print(line)
