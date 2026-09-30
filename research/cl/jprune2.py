"""Junk-track pruning with a per-movie cap: same model and break-even rule as jprune.apply, but at most max_frac of the movie's
nodes are removed (tracks with the lowest predicted annotated edges per node first). Limits the damage if the test movies'
annotation density differs from training."""
import numpy as np
from jprune import features


def apply(nodes, edges, model_path, lam=1.0, tp_ref=600., ratio=1.04, max_frac=0.10):
    import lgb_np
    trk, X = features(nodes, edges)
    if not trk: return nodes, edges, {'jp_removed_tracks': 0, 'jp_removed_nodes': 0, 'jp_capped': 0}
    m = lgb_np.load(model_path)
    pr = np.exp(m.raw(X)) if str(getattr(m, 'objective', '')).startswith('poisson') else m.predict(X)
    npred = len(nodes); ntot_hat = npred * ratio; m_hat = 1 - 0.1 * (npred - ntot_hat) / ntot_hat
    be = 0.1 * tp_ref / ntot_hat / m_hat
    lens = np.array([len(c) for c in trk], float)
    sel = pr < lam * lens * be
    capped = 0
    if max_frac is not None and lens[sel].sum() > max_frac * npred:
        idx = np.flatnonzero(sel); order = idx[np.argsort(pr[idx] / lens[idx], kind='stable')]
        keep = order[np.cumsum(lens[order]) <= max_frac * npred]
        sel = np.zeros_like(sel); sel[keep] = True; capped = 1
    rm = set(n for c, s in zip(trk, sel) if s for n in c)
    nodes = {n: v for n, v in nodes.items() if n not in rm}
    edges = [e for e in edges if int(e['source_id']) not in rm and int(e['target_id']) not in rm]
    return nodes, edges, {'jp_removed_tracks': int(sel.sum()), 'jp_removed_nodes': len(rm), 'jp_capped': capped}
