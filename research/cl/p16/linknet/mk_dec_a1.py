"""decisions for standalone linknet (LOEO scores), a-priori threshold exp(A1) >= TH, greedy one-touch per movie."""
import pickle, sys, numpy as np
TH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
R = pickle.load(open('/workspace/cl/p16/linknet/out/eval_v1_44b6_v1_6bba.pkl', 'rb')); dec = {}; n = 0
for emb, D in R.items():
    for d in D:
        p = np.exp(d['scv']['ln_A1']); o = np.argsort(-p); used = set(); out = []
        for j in o:
            if p[j] < TH: break
            ks = [x for x in (d['S'][j], d['D'][j], d['C'][j], d['Q'][j]) if x >= 0]
            if any(x in used for x in ks): continue
            used.update(ks); out.append(tuple(int(d['ids'][x]) if x >= 0 else -1 for x in (d['S'][j], d['D'][j], d['C'][j], d['Q'][j])))
        dec[d['movie']] = out; n += len(out)
pickle.dump(dec, open('/workspace/cl/p16/linknet/out/dec_v1_A1_%s.pkl' % TH, 'wb')); print('decisions', n)
