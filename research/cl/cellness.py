"""Node-level 'annotated-cell likeness' from b1 appearance embeddings; analysis of junk-removal potential."""
import glob, json, sys
import numpy as np
from pathlib import Path
from collections import defaultdict
import lightgbm as lgb

TR = ['t127a', 't127b', 'audit32']


def load(sets, kind_sel=None, max_neg_per_movie=None, seed=0):
    X, y, meta = [], [], []
    rng = np.random.default_rng(seed)
    for s in sets:
        for f in sorted(glob.glob('/workspace/cl/emb/%s__*.npz' % s)):
            d = np.load(f)
            k = d['kind']; m = np.ones(len(k), bool) if kind_sel is None else (k == kind_sel)
            lab = np.where(k == 0, d['matched'], (d['d_any'] <= 7).astype(np.int8))
            idx = np.where(m)[0]
            if max_neg_per_movie:
                pos = idx[lab[idx] == 1]; neg = idx[lab[idx] == 0]
                if len(neg) > max_neg_per_movie: neg = rng.choice(neg, max_neg_per_movie, replace=False)
                idx = np.concatenate([pos, neg])
            T = d['t'][idx]; Tm = max(1, d['t'].max())
            feats = np.concatenate([d['emb'][idx].astype(np.float32), d['pos'][idx][:, :1] * 1.625, (T / Tm)[:, None], k[idx][:, None]], 1)
            X.append(feats); y.append(lab[idx]); meta += [(s, Path(f).stem.split('__')[1])] * len(idx)
    return np.concatenate(X), np.concatenate(y), meta


if __name__ == '__main__':
    X, y, _ = load(TR, max_neg_per_movie=6000)
    print('train', X.shape, 'pos', y.sum(), flush=True)
    params = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=32)
    b = lgb.train(params, lgb.Dataset(X, y), 400)
    b.save_model('/workspace/cl/cell_lgb.txt'); json.dump(b.dump_model(), open('/workspace/cl/cell_lgb.json', 'w'))
    # evaluate on hold36 (all nodes)
    for s in ['hold36', 'prev4']:
        Xh, yh, meta = load([s])
        p = b.predict(Xh)
        kind = Xh[:, -1]
        o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(len(p)); n1 = yh.sum(); n0 = len(yh) - n1
        auc = (r[yh == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
        print(s, 'nodes', len(yh), 'pos', int(n1), 'AUC %.3f' % auc)
        for kd, nm in [(0, 'final'), (1, 'dropped')]:
            m = kind == kd
            base = yh[m].mean()
            qs = np.quantile(p[m], [0.05, 0.1, 0.2, 0.3, 0.5])
            line = '  %s n=%d base %.4f |' % (nm, m.sum(), base)
            for q, frac in zip(qs, [0.05, 0.1, 0.2, 0.3, 0.5]):
                mm = m & (p <= q); line += ' bottom%d%%: rate %.4f (rel %.3f)' % (int(frac * 100), yh[mm].mean(), yh[mm].mean() / base)
            print(line)
            if kd == 1:
                for th in [0.02, 0.05, 0.1, 0.2]:
                    mm = m & (p >= th); print('    dropped p>=%.2f: n %d near-GT rate %.4f' % (th, mm.sum(), yh[mm].mean() if mm.sum() else 0))
