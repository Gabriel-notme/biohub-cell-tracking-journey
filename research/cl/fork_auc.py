import json, numpy as np
from collections import defaultdict
R = json.load(open('/workspace/cl/forks2_p3.json'))
F = ['t','z','hist','la','lb','ea','eb','anyfork','d_near','d_far','d_ab','cos','dz_ab','vel','dens_p','dens_mid','nn_p','nfork','src_dc','src_dsr']
def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))
tr = [r for r in R if r['set'] in ('t127a', 't127b', 'audit32') and r['lab'] != 'U']
te = [r for r in R if r['set'] in ('hold36', 'prev4') and r['lab'] != 'U']
allc = [r for r in R if r['lab'] != 'U']
print('train TP/FP', sum(r['lab'] == 'TP' for r in tr), sum(r['lab'] == 'FP' for r in tr), ' test', sum(r['lab'] == 'TP' for r in te), sum(r['lab'] == 'FP' for r in te))
print('%-9s %7s %7s %7s | medTP  medFP  medU' % ('feat', 'AUCtr', 'AUCte', 'AUCall'))
U = [r for r in R if r['lab'] == 'U']
for f in F:
    a1 = auc([r[f] for r in tr if r['lab'] == 'TP'], [r[f] for r in tr if r['lab'] == 'FP'])
    a2 = auc([r[f] for r in te if r['lab'] == 'TP'], [r[f] for r in te if r['lab'] == 'FP'])
    a3 = auc([r[f] for r in allc if r['lab'] == 'TP'], [r[f] for r in allc if r['lab'] == 'FP'])
    print('%-9s %7.3f %7.3f %7.3f | %6.2f %6.2f %6.2f' % (f, a1, a2, a3, np.median([r[f] for r in allc if r['lab'] == 'TP']), np.median([r[f] for r in allc if r['lab'] == 'FP']), np.median([r[f] for r in U])))
# counted rate vs uncounted: what fraction of forks are counted by feature bins
