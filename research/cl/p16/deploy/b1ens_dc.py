"""P16 division-completion scorer: the deployed b1 fork head plus an ensemble of b1-recipe seeds trained on all labelled movies.
div_complete.score_candidates still scores every candidate with the deployed b1 (unchanged); then, per candidate type
(start / stolen), the top-K candidates by b1 are re-scored by the ensemble, and the choice order becomes the equal-weight mean
logit of all models (deployed b1 + each ensemble seed). Rank mapping: per movie and type, the i-th best candidate by the new
order receives b1's i-th best probability, so P15's thresholds accept exactly as many forks as P15 would; only which forks are
chosen can change. Any error falls back to the unchanged P15 scores."""
import numpy as np
from pathlib import Path

_ER = {}
STATS = {'b1ens_calls': 0, 'b1ens_rescored': 0, 'b1ens_errors': 0, 'b1ens_changed': 0}


def _logit(x):
    x = np.clip(np.asarray(x, np.float64), 1e-7, 1 - 1e-7)
    return np.log(x / (1 - x))


def install(dc, model_paths, K=300):
    if getattr(dc, '_b1ens_installed', False): return
    orig = dc.score_candidates
    paths = [Path(p) for p in model_paths]

    def patched(er, name, nodes, edges, with_edges=False):
        res = orig(er, name, nodes, edges, with_edges)
        if not res: return res
        STATS['b1ens_calls'] += 1
        try:
            from refine_events import EventRefiner
            from cell_event import SCALE, chain, fork_geometry
            if 'ens' not in _ER:
                _ER['ens'] = EventRefiner(paths, er.root, {**er.config, 'fork_ensemble': 'mean', 'ensemble': 'mean'}, Path(er.cache_dir) / 'b1ens')
            ens = _ER['ens']
            b1 = np.array([r['fork'] for r in res], np.float64)
            pick = []
            for typ in ('start', 'stolen'):
                idx = np.array([i for i, r in enumerate(res) if r['typ'] == typ], np.int64)
                if len(idx): pick += idx[np.argsort(-b1[idx], kind='stable')[:K]].tolist()
            pick = sorted(pick)
            if not pick: return res
            triples = [(res[i]['p'], res[i]['a'], res[i]['b']) for i in pick]
            out, prev = dc.structure(nodes, edges)
            pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
            fg = [fork_geometry(chain(p, prev, pos), chain(a, out, pos), chain(b, out, pos)) for p, a, b in triples]
            sub = {n: nodes[n] for n in {x for t in triples for x in t}}
            ids, emb = ens.embeddings(name, sub); lookup = {n: i for i, n in enumerate(ids)}
            pe = ens.score('fork', triples, fg, emb, lookup)  # sigmoid of the mean logit over the ensemble seeds
            m = len(paths)
            comb = (_logit(b1[pick]) + m * _logit(pe)) / (m + 1)
            new = -1e6 + b1  # candidates outside the shortlist keep their b1 order below every shortlisted one
            new[pick] = comb
            out_res = [dict(r) for r in res]
            for typ in ('start', 'stolen'):
                idx = [i for i, r in enumerate(res) if r['typ'] == typ]
                if not idx: continue
                b1s = sorted((b1[i] for i in idx), reverse=True)
                for rank, i in enumerate(sorted(idx, key=lambda i: -new[i])):
                    out_res[i]['fork'] = float(b1s[rank])
            STATS['b1ens_rescored'] += len(pick)
            STATS['b1ens_changed'] += sum(1 for r0, r1 in zip(res, out_res) if r0['fork'] != r1['fork'])
            return out_res
        except Exception:
            STATS['b1ens_errors'] += 1
            return res

    dc.score_candidates = patched; dc._b1ens_installed = True
    import atexit, json
    atexit.register(lambda: print('B1ENS_STATS', json.dumps(STATS), flush=True))
