"""Dependency-free predictor for LightGBM binary models exported with Booster.dump_model() (JSON)."""
import json
import numpy as np


class NumpyGBM:
    def __init__(self, path):
        d = json.load(open(path))
        self.trees = [self._flatten(t['tree_structure']) for t in d['tree_info']]
        self.objective = d.get('objective', 'binary')

    @staticmethod
    def _flatten(root):
        feat, thr, left, right, val, dleft, nan_missing = [], [], [], [], [], [], []

        def add(node):
            i = len(feat)
            feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1); val.append(0.0); dleft.append(True); nan_missing.append(False)
            if 'leaf_value' in node:
                val[i] = float(node['leaf_value'])
                return i
            assert node.get('decision_type', '<=') == '<=', node.get('decision_type')
            feat[i] = int(node['split_feature']); thr[i] = float(node['threshold'])
            dleft[i] = bool(node.get('default_left', True)); nan_missing[i] = node.get('missing_type', 'None') == 'NaN'
            l = add(node['left_child']); r = add(node['right_child'])
            left[i] = l; right[i] = r
            return i
        add(root)
        return (np.array(feat), np.array(thr), np.array(left), np.array(right), np.array(val), np.array(dleft), np.array(nan_missing))

    def raw(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros(len(X))
        rows = np.arange(len(X))
        for feat, thr, left, right, val, dleft, nanm in self.trees:
            idx = np.zeros(len(X), dtype=np.int64)
            while True:
                f = feat[idx]
                internal = f >= 0
                if not internal.any(): break
                ii = rows[internal]; ni = idx[internal]
                xv = X[ii, f[internal]]
                isnan = np.isnan(xv)
                go_left = np.where(isnan & nanm[ni], dleft[ni], xv <= thr[ni])
                idx[internal] = np.where(go_left, left[ni], right[ni])
            out += val[idx]
        return out

    def predict(self, X):
        r = self.raw(X)
        return 1.0 / (1.0 + np.exp(-r)) if self.objective.startswith('binary') else r


def load(path):
    path = str(path)
    if path.endswith('.json'):
        return NumpyGBM(path)
    import lightgbm as lgb
    return lgb.Booster(model_file=path)
