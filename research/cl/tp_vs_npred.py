"""Does the number of annotated (TP) edges per movie scale with the number of predicted nodes? (train pkls only)"""
import pickle
import numpy as np
TR = [r for t in ['p8_t127a', 'p8_t127b', 'p8_audit32'] for r in pickle.load(open('/workspace/cl/jp/%s.pkl' % t, 'rb'))]
tp = np.array([r['TP'] for r in TR], float); npred = np.array([r['npred'] for r in TR], float); emb = np.array([r['movie'][:4] for r in TR])
print('corr(log TP, log npred) %.3f' % np.corrcoef(np.log(tp + 1), np.log(npred))[0, 1])
b, a = np.polyfit(np.log(npred), np.log(tp + 1), 1); print('log-log fit: TP ~ %.3g * npred^%.3f' % (np.exp(a), b))
for e in ['44b6', '6bba']:
    k = emb == e; print(e, 'n %d median TP %.0f median npred %.0f corr %.3f' % (k.sum(), np.median(tp[k]), np.median(npred[k]), np.corrcoef(np.log(tp[k] + 1), np.log(npred[k]))[0, 1]))
for lo, hi in [(0, 10000), (10000, 20000), (20000, 30000), (30000, 50000), (50000, 1e9)]:
    k = (npred >= lo) & (npred < hi)
    if k.sum(): print('npred %6d-%6d: n %3d TP median %.0f  TP/npred median %.4f' % (lo, min(hi, 99999), k.sum(), np.median(tp[k]), np.median(tp[k] / npred[k])))
