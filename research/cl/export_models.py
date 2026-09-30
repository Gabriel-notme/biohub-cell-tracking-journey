import json, glob, sys
sys.path.insert(0, '/workspace/cl')
import numpy as np, lightgbm as lgb
import lgb_np, edge_link, relink
for name, feats, cdir in [('edge_lgb', edge_link.FEATS, '/workspace/cl/ecands'), ('relink_lgb', relink.FEATS, '/workspace/cl/rcands')]:
    b = lgb.Booster(model_file='/workspace/cl/%s.txt' % name)
    json.dump(b.dump_model(), open('/workspace/cl/%s.json' % name, 'w'))
    rows = []
    for f in sorted(glob.glob(cdir + '/hold36__*.json'))[:12]: rows += json.load(open(f))
    X = np.array([[(-1 if r.get(k) is None else r.get(k, -1)) for k in feats] for r in rows], dtype=np.float32)
    p1 = b.predict(X); p2 = lgb_np.load('/workspace/cl/%s.json' % name).predict(X)
    print(name, len(rows), 'max abs diff %.2e' % np.abs(p1 - p2).max(), 'n>=0.4 %d %d' % ((p1 >= .4).sum(), (p2 >= .4).sum()))
