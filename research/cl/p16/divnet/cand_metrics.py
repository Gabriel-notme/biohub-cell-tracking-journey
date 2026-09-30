"""Candidate-level metrics on the P15 div_complete pool rows (and GT rows) for LOEO predictions.
usage: cand_metrics.py '<json {label: {"44b6": pred_npz_for_44b6_movies, "6bba": pred_npz_for_6bba_movies}}>' [subset: all|clean40]
Rows: cand P=1 vs N/X=0 (N/X weighted by inverse sampling rate; D = already-recovered duplicate, reported separately),
existing forks Ftp vs Ffp (veto), GT rows G1 vs G0. Per test embryo, start and stolen separately."""
import sys, json, glob, os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
D = '/dev/shm/divnet'
arms = json.loads(sys.argv[1]); subset = sys.argv[2] if len(sys.argv) > 2 else 'all'
names = sorted(os.path.basename(f)[:-4] for f in glob.glob(D + '/*.npz'))
if subset == 'clean40':
    cl = set(open('/workspace/hold36.txt').read().split()) | set(open('/workspace/preview4.txt').read().split())
    names = [n for n in names if n in cl]
meta = {}
for n in names:
    d = np.load(D + '/' + n + '.npz'); meta[n] = {k: d[k] for k in ['y', 'w', 'typ', 'src', 'lab']}


def wprec_at(y, w, s, nsel):
    """weighted precision/recall when accepting the top candidates whose weighted count sums to nsel"""
    o = np.argsort(-s); cw = np.cumsum(w[o]); k = np.searchsorted(cw, nsel) + 1
    sel = o[:k]; return float(y[sel].sum() / w[sel].sum()), int(y[sel].sum())


for lab, spec in arms.items():
    print('==', lab, subset)
    for emb in ['44b6', '6bba']:
        P = np.load(spec[emb]); mv = [n for n in names if n.startswith(emb)]
        cols = {k: np.concatenate([meta[n][k] for n in mv]) for k in ['y', 'w', 'typ', 'src', 'lab']}
        s = np.concatenate([P[n] for n in mv])
        line = '  %s |' % emb
        for typ, tname in [(0, 'start'), (1, 'stolen')]:
            m = (cols['src'] == 'cand') & (cols['typ'] == typ) & (cols['y'] >= 0) & np.isfinite(s)
            y, w, ss = cols['y'][m].astype(float), cols['w'][m], s[m]
            if y.sum() == 0 or y.sum() == len(y): line += ' %s n/a |' % tname; continue
            auc = roc_auc_score(y, ss, sample_weight=w); ap = average_precision_score(y, ss, sample_weight=w)
            npos = int(y.sum()); tot = w.sum()
            pk = [wprec_at(y, w, ss, k) for k in (npos, 3 * npos)]
            dm = (cols['src'] == 'cand') & (cols['typ'] == typ) & (cols['y'] < 0) & np.isfinite(s)
            # fraction of duplicates (D) scoring above the median positive
            dup = float((s[dm] > np.median(ss[y > 0])).mean()) if dm.any() else float('nan')
            line += ' %s P=%d N~%d AUC %.3f AP %.4f prec@P %.3f(%d) prec@3P %.3f(%d) Dhi %.2f |' % (tname, npos, tot - npos, auc, ap, pk[0][0], pk[0][1], pk[1][0], pk[1][1], dup)
        m = (cols['src'] == 'fork') & np.isfinite(s)
        if m.any() and 0 < cols['y'][m].sum() < m.sum():
            line += ' veto Ftp/Ffp=%d/%d AUC %.3f |' % (cols['y'][m].sum(), (cols['y'][m] == 0).sum(), roc_auc_score(cols['y'][m], s[m]))
        m = (cols['src'] == 'gt') & np.isfinite(s)
        if m.any() and 0 < cols['y'][m].sum() < m.sum():
            line += ' GT AUC %.3f AP %.3f' % (roc_auc_score(cols['y'][m], s[m]), average_precision_score(cols['y'][m], s[m]))
        print(line)
