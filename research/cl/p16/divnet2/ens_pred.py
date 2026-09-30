"""Average the per-seed LOEO row predictions of a variant -> models/<V>ens_tr<E>_pred.npz (for cand_metrics.py)."""
import sys, glob, numpy as np
V = sys.argv[1]
for E in ['44b6', '6bba']:
    fs = sorted(glob.glob('/workspace/cl/p16/divnet2/models/%s_tr%s_s[0-9]_pred.npz' % (V, E))); assert len(fs) == 3, fs
    P = [np.load(f) for f in fs]
    np.savez('/workspace/cl/p16/divnet2/models/%sens_tr%s_pred.npz' % (V, E), **{k: np.mean([p[k] for p in P], 0) for k in P[0].files})
    print(V, E, len(fs))
