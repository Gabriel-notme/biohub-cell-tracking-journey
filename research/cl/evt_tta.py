"""Test-time augmentation for B5's b1/b2 cell-event models (EventRefiner), installed as a monkeypatch.
The models were trained with random z/y/x flips and a y<->x transpose, with the edge geometry transformed the same way
(train_events.augment: negate component j of delta, v, delta-v, w per flipped axis; swap y/x components on transpose; fork
geometry is rotation-invariant and left as is). Here every node patch is encoded under each view, each view's edge/fork logits
are computed with the matching geometry, and the logits are averaged over views before B5's usual model-ensemble rule.
Callers that read the embedding banks directly (e.g. visual_edge features) still get the identity view, unchanged.
Mode from env EVT_TTA: none | flip4 (y/x flips) | d4 (y/x flips x transpose, 8 views) | full (z too, 16 views)."""
import os, json, hashlib
import numpy as np
import torch

_VIEWS = {}


def views(mode):
    if mode == 'flip4': return [(0, fy, fx, 0) for fy in (0, 1) for fx in (0, 1)]
    if mode == 'd4': return [(0, fy, fx, tr) for tr in (0, 1) for fy in (0, 1) for fx in (0, 1)]
    if mode == 'full': return [(fz, fy, fx, tr) for fz in (0, 1) for tr in (0, 1) for fy in (0, 1) for fx in (0, 1)]
    return [(0, 0, 0, 0)]


def tx(x, v):
    fz, fy, fx, tr = v
    dims = [d for d, f in ((-3, fz), (-2, fy), (-1, fx)) if f]
    if dims: x = x.flip(dims)
    if tr: x = x.transpose(-1, -2)
    return x


def tg(g, v):
    g = np.array(g, np.float32, copy=True); fz, fy, fx, tr = v
    for j, f in ((0, fz), (1, fy), (2, fx)):
        if f:
            for s in (0, 4, 7, 10): g[:, s + j] *= -1
    if tr:
        for s in (0, 4, 7, 10): g[:, [s + 1, s + 2]] = g[:, [s + 2, s + 1]]
    return g


def install():
    import refine_events
    from refine_events import EventRefiner
    from cell_event import Movie
    if getattr(EventRefiner, '_tta_installed', False): return
    orig_emb, orig_score = EventRefiner.embeddings, EventRefiner.score

    @torch.inference_mode()
    def embeddings(self, name, nodes):
        mode = os.environ.get('EVT_TTA', 'none'); V = views(mode)
        if len(V) == 1: return orig_emb(self, name, nodes)
        ids = sorted(nodes, key=lambda n: (nodes[n]['t'], n))
        signature = hashlib.sha256(json.dumps([(n, nodes[n]['t'], nodes[n]['z'], nodes[n]['y'], nodes[n]['x']) for n in ids]).encode()).hexdigest()[:16]
        model_signature = hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths) + str(self.amp_dtype).encode()).hexdigest()[:16]
        path = self.cache_dir / (name + '_' + signature + '_' + model_signature + '_tta' + mode + '.npz')
        if path.exists():
            d = np.load(path); allv = [d['m' + str(i)] for i in range(len(self.models))]
        else:
            movie = Movie(self.root / (name + '.zarr'), context=max(5, max(m.context for m in self.models))); acc = [[] for _ in self.models]
            for start in range(0, len(ids), 192):
                chosen = ids[start:start + 192]
                x = movie.patches([nodes[n]['t'] for n in chosen], [[nodes[n][k] for k in ['z', 'y', 'x']] for n in chosen])
                x = torch.from_numpy(x).to(self.device, dtype=torch.float32)
                for j, model in enumerate(self.models):
                    out = []
                    for v in V:
                        with torch.autocast('cuda', dtype=self.amp_dtype, enabled=self.device == 'cuda'):
                            emb = model.encode(tx(x, v).contiguous())
                        out.append(emb.float().cpu().numpy())
                    acc[j].append(np.stack(out))
            allv = [np.concatenate(r, axis=1) for r in acc]
            tmp = path.with_name(path.stem + '.' + str(os.getpid()) + '.tmp.npz')
            np.savez_compressed(tmp, **{'m' + str(i): r for i, r in enumerate(allv)}); os.replace(tmp, path)
        banks = [np.ascontiguousarray(r[0]) for r in allv]
        if len(_VIEWS) > 64: _VIEWS.clear()
        for b, r in zip(banks, allv): _VIEWS[id(b)] = (b, r, mode)
        return ids, banks

    @torch.inference_mode()
    def score(self, kind, rows, geometry, embeddings, lookup):
        tt = [_VIEWS.get(id(b)) for b in embeddings]
        if not rows or any(t is None or t[0] is not b for t, b in zip(tt, embeddings)):
            return orig_score(self, kind, rows, geometry, embeddings, lookup)
        V = views(tt[0][2])
        indices = np.asarray([[lookup[n] for n in r] for r in rows], np.int64)
        geometry = np.asarray(geometry, np.float32); results = []
        for model, (b, allv, _) in zip(self.models, tt):
            acc = np.zeros(len(rows), np.float64)
            for vi, v in enumerate(V):
                gv = tg(geometry, v) if kind == 'edge' else geometry; bank = allv[vi]; chunks = []
                for start in range(0, len(rows), 2048):
                    ix = indices[start:start + 2048]; g = torch.from_numpy(np.ascontiguousarray(gv[start:start + 2048])).to(self.device)
                    z = [torch.from_numpy(bank[ix[:, i]]).to(self.device) for i in range(ix.shape[1])]
                    logits = model.edge_logits(*z, g) if kind == 'edge' else model.fork_logits(*z, g)
                    chunks.append(logits.float().cpu().numpy())
                acc += np.concatenate(chunks)
            results.append((acc / len(V)).astype(np.float32))
        stacked = np.stack(results); logits = stacked.mean(0)
        mode = self.config.get(kind + '_ensemble', self.config['ensemble'])
        if mode == 'conservative': logits = .65 * logits + .35 * stacked.min(0)
        elif mode == 'last': logits = stacked[-1]
        elif mode == 'first': logits = stacked[0]
        elif mode == 'max_fork' and kind == 'fork': logits = stacked.max(0)
        return 1 / (1 + np.exp(-np.clip(logits / self.config['temperature'], -30, 30)))

    EventRefiner.embeddings = embeddings; EventRefiner.score = score; EventRefiner._tta_installed = True
