"""b1ens copy of divnet/dn_patch.py: DN_B1LOEO glob may match several checkpoints (mean fork logit).
Monkeypatch for P-stage division completion: replace div_complete.score_candidates' fork probability with a
leave-one-embryo-out scorer (movie 44b6_* scored by models trained on 6bba only, and vice versa).
env DN_MODE: none | divnet | b1loeo ; DN_MODELS: glob with {E} = training embryo (divnet: *.pt DivNet checkpoints, averaged);
DN_B1LOEO: glob with {E} for the LOEO b1-recipe checkpoint (mode b1loeo, and the b1loeo shortlist);
DN_SHORT (divnet only): '<b1dep|b1loeo>:K' = DivNet scores only the top-K candidates per type by that scorer (a movie has ~1e5
candidates; one DivNet crop each is infeasible), the rest keep their shortlist order below every shortlisted candidate;
DN_MAP: rank (per movie and type, the i-th best new-score candidate receives b1dep's i-th best probability, so the P15 thresholds
accept the same number as P15 would) | raw ; DN_LOG: directory for per-movie candidate logs (optional)."""
import os, sys, json, glob
import numpy as np
import torch
sys.path.insert(0, '/workspace/cl/p16/divnet')
_M = {}; _FC = {}; _ER = {}


def _divnet_models(emb):
    if emb not in _M:
        import divnet_lib as L
        ms = []
        for p in sorted(glob.glob(os.environ['DN_MODELS'].replace('{E}', emb))):
            ck = torch.load(p, map_location='cpu', weights_only=False)
            m = L.DivNet(**ck['config']); m.load_state_dict(ck['model']); ms.append(m.cuda().eval())
        assert ms, 'no DivNet models for ' + emb
        _M[emb] = ms
    return _M[emb]


@torch.inference_mode()
def divnet_scores(zpath, nodes, triples, emb):
    import divnet_lib as L
    if _FC.get('path') != str(zpath):
        arr, low, high = L.load_movie(zpath); _FC.clear(); _FC['path'] = str(zpath); _FC['fc'] = L.FrameCache(arr, low, high)
    fc = _FC['fc']; ms = _divnet_models(emb); acc = np.zeros(len(triples), np.float64)
    for i0 in range(0, len(triples), 1024):
        rows = [(int(nodes[p]['t']), [float(nodes[p][k]) for k in 'zyx'], [float(nodes[a][k]) for k in 'zyx'], [float(nodes[b][k]) for k in 'zyx']) for p, a, b in triples[i0:i0 + 1024]]
        X, rel = L.crops(fc, rows); X = torch.from_numpy(X).cuda(); rel = torch.from_numpy(rel).cuda()
        for m in ms:
            for i in range(0, len(X), 128):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    acc[i0 + i:i0 + i + 128] += m(X[i:i + 128], rel[i:i + 128]).float().sigmoid().cpu().numpy()
    return acc / len(ms)


def install(dc):
    if getattr(dc, '_dn_installed', False): return
    orig = dc.score_candidates
    mode = os.environ.get('DN_MODE', 'divnet'); mapping = os.environ.get('DN_MAP', 'rank'); logdir = os.environ.get('DN_LOG')
    if mode == 'none': return
    short = os.environ.get('DN_SHORT', 'b1loeo:300'); calls = {}

    def b1loeo(er, name, nodes, edges, other, triples):
        from refine_events import EventRefiner
        if other not in _ER:
            ck = sorted(glob.glob(os.environ['DN_B1LOEO'].replace('{E}', other))); assert len(ck) >= 1, ck; print('B1E_LOADED', other, len(ck), ck, flush=True)
            _ER[other] = EventRefiner(ck, er.root, {**er.config, 'fork_ensemble': 'mean'}, er.cache_dir / ('b1loeo_tr' + other))  # mean fork logit over ckpts
        r2 = orig(_ER[other], name, nodes, edges, False)
        assert [(r['p'], r['a'], r['b']) for r in r2] == triples
        return np.array([r['fork'] for r in r2])

    def patched(er, name, nodes, edges, with_edges=False):
        res = orig(er, name, nodes, edges, with_edges)
        if not res: return res
        emb = name[:4]; other = {'44b6': '6bba', '6bba': '44b6'}[emb]
        triples = [(r['p'], r['a'], r['b']) for r in res]
        b1 = np.array([r['fork'] for r in res]); aux = np.full(len(res), np.nan)
        if mode == 'b1loeo':
            new = b1loeo(er, name, nodes, edges, other, triples)
        else:
            src, K = short.split(':'); K = int(K)
            sl = b1 if src == 'b1dep' else b1loeo(er, name, nodes, edges, other, triples)
            aux = sl; new = -1 + 1e-3 * sl; pick = []
            for typ in ('start', 'stolen'):
                idx = np.array([i for i, r in enumerate(res) if r['typ'] == typ], np.int64)
                if len(idx): pick += idx[np.argsort(-sl[idx])[:K]].tolist()
            pick = sorted(pick)
            if pick:
                dn = divnet_scores(er.root / (name + '.zarr'), nodes, [triples[i] for i in pick], other)
                if os.environ.get('DN_COMBINE') == 'logit_mean':  # pre-registered: equal-weight logit mean with b1dep
                    lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
                    dn = 1 / (1 + np.exp(-(0.5 * lg(b1[pick]) + 0.5 * lg(dn))))
                new[pick] = dn
        for r, x, y in zip(res, new, aux): r['fork_b1'] = r['fork']; r['fork_new'] = float(x); r['fork_aux'] = float(y)
        if mapping == 'rank':
            for typ in ('start', 'stolen'):
                idx = [i for i, r in enumerate(res) if r['typ'] == typ]
                if not idx: continue
                b1s = sorted((res[i]['fork_b1'] for i in idx), reverse=True)
                for rank, i in enumerate(sorted(idx, key=lambda i: -res[i]['fork_new'])): res[i]['fork'] = b1s[rank]
        else:
            for r in res: r['fork'] = r['fork_new']
        if logdir:
            k = calls[name] = calls.get(name, 0) + 1
            os.makedirs(logdir, exist_ok=True)
            keep = [r for r in res if r['fork_b1'] >= 0.01 or r['fork_new'] > -0.5 or (r['fork_aux'] == r['fork_aux'] and r['fork_aux'] >= 0.01)]
            with open(os.path.join(logdir, '%s_%d.json' % (name, k)), 'w') as f:
                json.dump([{x: r[x] for x in ('p', 'a', 'b', 'q', 'typ', 'fork', 'fork_b1', 'fork_new', 'fork_aux')} for r in keep], f)
        if torch.cuda.is_available(): torch.cuda.empty_cache()  # b1ens: release cached GPU memory between movies (many concurrent workers)
        return res

    dc.score_candidates = patched; dc._dn_installed = True
