"""P16 (experimental, NOT strict-validated): DivNet second opinion inside P15's division completion.
div_complete.score_candidates still scores every candidate with the deployed b1; the top-K candidates per type (start / stolen) by b1
are re-scored by DivNet (event-centred 3D CNN, 3 seeds trained on all 199 labelled movies; divnet_core.py). The new choice order is
  mode 'blend'  : sigmoid(0.5 logit(b1) + 0.5 logit(DivNet))   (pre-registered 'dnC' form)
  mode 'rerank' : DivNet alone within b1's top-K                ('dnB' form)
and is rank-mapped onto b1's sorted probabilities per movie and type, so P15's thresholds accept exactly as many forks as P15;
only which forks are chosen can change. Any error falls back to P15's unchanged scores."""
import atexit, json
import numpy as np

STATS = {'dn_calls': 0, 'dn_rescored': 0, 'dn_changed': 0, 'dn_errors': 0, 'dn_last_error': ''}


def _lg(x):
    x = np.clip(np.asarray(x, np.float64), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def rescore(scorer, zarr_path, nodes, res, K=100, mode='blend'):
    b1 = np.array([r['fork'] for r in res], np.float64)
    new = np.zeros(len(res)); pick = []
    for typ in ('start', 'stolen'):
        idx = np.array([i for i, r in enumerate(res) if r['typ'] == typ], np.int64)
        if len(idx): pick += idx[np.argsort(-b1[idx], kind='stable')[:K]].tolist()
    pick = sorted(pick)
    if not pick: return res
    dn = scorer.score(zarr_path, nodes, [(res[i]['p'], res[i]['a'], res[i]['b']) for i in pick])
    new[pick] = 0.5 * _lg(b1[pick]) + 0.5 * _lg(dn) if mode == 'blend' else _lg(dn)
    out = [dict(r) for r in res]; ps = set(pick)
    # shortlisted candidates first (by the new score), then the rest in their original b1 order
    key = lambda i: (0, -new[i], i) if i in ps else (1, -b1[i], i)
    for typ in ('start', 'stolen'):
        idx = [i for i, r in enumerate(res) if r['typ'] == typ]
        if not idx: continue
        b1s = sorted((b1[i] for i in idx), reverse=True)
        for rank, i in enumerate(sorted(idx, key=key)): out[i]['fork'] = float(b1s[rank])
    STATS['dn_rescored'] += len(pick)
    STATS['dn_changed'] += sum(1 for r0, r1 in zip(res, out) if r0['fork'] != r1['fork'])
    return out


def install(dc, weights, K=100, mode='blend'):
    if getattr(dc, '_divnet16_installed', False): return
    import divnet_core
    orig = dc.score_candidates
    scorer = divnet_core.DivNetScorer([str(w) for w in weights])

    def patched(er, name, nodes, edges, with_edges=False):
        res = orig(er, name, nodes, edges, with_edges)
        if not res: return res
        STATS['dn_calls'] += 1
        try:
            return rescore(scorer, er.root / (name + '.zarr'), nodes, res, K=K, mode=mode)
        except Exception as e:
            STATS['dn_errors'] += 1; STATS['dn_last_error'] = repr(e)[:300]
            return res

    dc.score_candidates = patched; dc._divnet16_installed = True
    atexit.register(lambda: print('DIVNET16_STATS', json.dumps(STATS), flush=True))
