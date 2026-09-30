"""DivNet: event-centred 3D CNN fork (division) scorer, self-contained deployable module.

A candidate division is a triple (p at t, a and b at t+1) of graph nodes (dicts with t, z, y, x in full-resolution voxels).
For each triple one crop is cut from the movie: frames t-2..t+2 of a (16, 48, 48) box on the (1,2,2)-strided grid (26 x 39 x 39 um),
centred between p and the a/b midpoint; two marker channels (Gaussian at p; Gaussians at a and b) tell the network which cells
are meant. DivNet(5 frames + 2 markers) -> logit.  Weights: dn1_full_trall_s{0,1,2}.pt (all 199 movies, deploy);
dn1_full_tr{44b6,6bba}_s{0,1,2}.pt (leave-one-embryo-out versions used for validation).

Use inside the P-stage (div_complete.score_candidates), the validated form ('dnC'):
    import divnet
    divnet.install(dc, ['.../dn1_full_trall_s0.pt', '.../dn1_full_trall_s1.pt', '.../dn1_full_trall_s2.pt'], K=300)
right after `import div_complete as dc` in the P-stage worker. The wrapper keeps b1's candidate list and probabilities, re-scores the
top-K candidates per type (start / stolen) by b1 with DivNet, combines sigmoid(0.5 logit(b1) + 0.5 logit(DivNet)), and maps the new
ranking back onto b1's sorted probabilities per movie and type ('rank mapping'), so dc.apply's thresholds (0.9 / 0.97) accept the
same number of candidates as before; only which candidates are accepted changes.
"""
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STRIDE = np.array([1., 2., 2.], np.float32)
CROP = (16, 48, 48)
DT = (-2, -1, 0, 1, 2)


def load_movie(path):
    try:
        import numcodecs.blosc; numcodecs.blosc.use_threads = False
    except Exception:
        pass
    import zarr
    g = zarr.open_group(str(path), mode='r'); arr = g['0']
    qs = g.attrs.get('image_statistics', {}).get('quantiles', {})
    low = float(qs.get('0.001', 0)); high = float(qs.get('0.999', 0))
    if high <= low:
        v = np.asarray(arr[0])[::2, ::4, ::4]; low, high = np.percentile(v, [.1, 99.9])
    return arr, low, high


class FrameCache:
    def __init__(self, arr, low, high):
        self.arr, self.low, self.high = arr, low, high; self.T = arr.shape[0]; self.c = {}
    def get(self, t):
        t = int(min(max(t, 0), self.T - 1))
        if t not in self.c:
            self.c[t] = np.clip((np.asarray(self.arr[t, :, ::2, ::2], np.float32) - self.low) / (self.high - self.low + 1e-6), 0, 3).astype(np.float16)
        return self.c[t]


def crops(fc, rows):
    """rows: list of (t, p_zyx, a_zyx, b_zyx) in full-resolution voxel coords -> X [R,5,16,48,48] fp16, rel [R,3,3] fp32."""
    R = len(rows); X = np.empty((R, len(DT)) + CROP, np.float16); rel = np.empty((R, 3, 3), np.float32)
    half = np.array(CROP) // 2; byt = {}
    for i, r in enumerate(rows): byt.setdefault(int(r[0]), []).append(i)
    for t in sorted(byt):
        fr = [fc.get(t + d) for d in DT]; shp = fr[0].shape
        for i in byt[t]:
            _, p, a, b = rows[i]
            ps = np.asarray(p, np.float32) / STRIDE; as_ = np.asarray(a, np.float32) / STRIDE; bs = np.asarray(b, np.float32) / STRIDE
            c = np.rint((ps + (as_ + bs) / 2) / 2).astype(int); lo = c - half
            ix = np.ix_(*[np.clip(np.arange(l, l + n), 0, s - 1) for l, n, s in zip(lo, CROP, shp)])
            for k in range(len(DT)): X[i, k] = fr[k][ix]
            rel[i] = np.stack([ps - lo, as_ - lo, bs - lo])
    return X, rel


def markers(rel, shape=CROP, sig=(1.2, 2.5, 2.5)):
    dev = rel.device
    zz = torch.arange(shape[0], device=dev, dtype=torch.float32).view(1, -1, 1, 1)
    yy = torch.arange(shape[1], device=dev, dtype=torch.float32).view(1, 1, -1, 1)
    xx = torch.arange(shape[2], device=dev, dtype=torch.float32).view(1, 1, 1, -1)
    def g(c):
        return torch.exp(-0.5 * (((zz - c[:, 0].view(-1, 1, 1, 1)) / sig[0]) ** 2 + ((yy - c[:, 1].view(-1, 1, 1, 1)) / sig[1]) ** 2 + ((xx - c[:, 2].view(-1, 1, 1, 1)) / sig[2]) ** 2))
    return torch.stack([g(rel[:, 0]), g(rel[:, 1]) + g(rel[:, 2])], 1)


class Res(nn.Module):
    def __init__(self, c):
        super().__init__(); self.net = nn.Sequential(nn.Conv3d(c, c, 3, padding=1, bias=False), nn.GroupNorm(8, c), nn.SiLU(), nn.Conv3d(c, c, 3, padding=1, bias=False), nn.GroupNorm(8, c))
    def forward(self, x): return F.silu(x + self.net(x))


def block(i, o, s):
    return nn.Sequential(nn.Conv3d(i, o, 3, stride=s, padding=1, bias=False), nn.GroupNorm(8, o), nn.SiLU(), Res(o))


class DivNet(nn.Module):
    def __init__(self, nframes=5, width=32, drop=0.3):
        super().__init__(); w = width; self.nframes = nframes
        self.body = nn.Sequential(block(nframes + 2, w, 1), block(w, 2 * w, (1, 2, 2)), block(2 * w, 4 * w, 2), block(4 * w, 6 * w, 2))
        self.head = nn.Sequential(nn.Linear(12 * w, 128), nn.SiLU(), nn.Dropout(drop), nn.Linear(128, 1))
    def forward(self, x, rel):
        x = x.float()
        if self.nframes != x.shape[1]:
            o = (x.shape[1] - self.nframes) // 2; x = x[:, o:o + self.nframes]
        m = x.mean((1, 2, 3, 4), keepdim=True); s = x.std((1, 2, 3, 4), keepdim=True).clamp_min(.03)
        x = ((x - m) / s).clamp(-5, 10)
        h = self.body(torch.cat([x, 2 * markers(rel, tuple(x.shape[-3:]))], 1))
        h = torch.cat([h.mean((2, 3, 4)), h.amax((2, 3, 4))], 1)
        return self.head(h).squeeze(-1)


class DivNetScorer:
    """Averaged probability of several DivNet checkpoints. amp: 'bf16' (validation runs) or 'fp16' (T4)."""
    def __init__(self, weights, device=None, amp=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if amp is None: amp = 'bf16' if (self.device == 'cuda' and torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8) else 'fp16'
        self.dtype = torch.bfloat16 if amp == 'bf16' else torch.float16
        self.models = []
        for p in weights:
            ck = torch.load(p, map_location='cpu', weights_only=False)
            m = DivNet(**ck['config']); m.load_state_dict(ck['model']); self.models.append(m.to(self.device).eval())
        self._fc = (None, None)

    @torch.inference_mode()
    def score(self, zarr_path, nodes, triples, batch=128):
        if not triples: return np.zeros(0)
        if self._fc[0] != str(zarr_path): self._fc = (str(zarr_path), FrameCache(*load_movie(zarr_path)))
        fc = self._fc[1]; acc = np.zeros(len(triples), np.float64)
        for i0 in range(0, len(triples), 1024):
            rows = [(int(nodes[p]['t']), [float(nodes[p][k]) for k in 'zyx'], [float(nodes[a][k]) for k in 'zyx'], [float(nodes[b][k]) for k in 'zyx']) for p, a, b in triples[i0:i0 + 1024]]
            X, rel = crops(fc, rows); X = torch.from_numpy(X).to(self.device); rel = torch.from_numpy(rel).to(self.device)
            for m in self.models:
                for i in range(0, len(X), batch):
                    with torch.autocast('cuda', dtype=self.dtype, enabled=self.device == 'cuda'):
                        acc[i0 + i:i0 + i + batch] += m(X[i:i + batch], rel[i:i + batch]).float().sigmoid().cpu().numpy()
        return acc / len(self.models)


def rescore(scorer, zarr_path, nodes, res, K=300):
    """res: div_complete.score_candidates output (list of dicts with p, a, b, typ, fork=b1 prob). Returns res with 'fork' replaced by the
    rank-mapped combined score (and 'fork_b1', 'fork_new' kept for logging)."""
    if not res: return res
    lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
    b1 = np.array([r['fork'] for r in res], np.float64); new = -1 + 1e-3 * b1; pick = []
    for typ in ('start', 'stolen'):
        idx = np.array([i for i, r in enumerate(res) if r['typ'] == typ], np.int64)
        if len(idx): pick += idx[np.argsort(-b1[idx])[:K]].tolist()
    pick = sorted(pick)
    if pick:
        dn = scorer.score(zarr_path, nodes, [(res[i]['p'], res[i]['a'], res[i]['b']) for i in pick])
        new[pick] = 1 / (1 + np.exp(-(0.5 * lg(b1[pick]) + 0.5 * lg(dn))))
    for r, x in zip(res, new): r['fork_b1'] = r['fork']; r['fork_new'] = float(x)
    for typ in ('start', 'stolen'):
        idx = [i for i, r in enumerate(res) if r['typ'] == typ]
        if not idx: continue
        b1s = sorted((res[i]['fork_b1'] for i in idx), reverse=True)
        for rank, i in enumerate(sorted(idx, key=lambda i: -res[i]['fork_new'])): res[i]['fork'] = b1s[rank]
    return res


def install(dc, weights, K=300, amp=None):
    """Monkeypatch div_complete.score_candidates (module object `dc`) in the P-stage worker process."""
    if getattr(dc, '_divnet_installed', False): return
    orig = dc.score_candidates; scorer = DivNetScorer(weights, amp=amp)

    def patched(er, name, nodes, edges, with_edges=False):
        res = orig(er, name, nodes, edges, with_edges)
        return rescore(scorer, er.root / (name + '.zarr'), nodes, res, K=K)
    dc.score_candidates = patched; dc._divnet_installed = True
