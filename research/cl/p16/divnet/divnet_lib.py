"""DivNet: event-centred 3D CNN fork scorer. One crop per candidate triple (parent p at t, daughters a, b at t+1):
5 frames (t-2..t+2) of a (16,48,48) crop on the (1,2,2)-strided grid (26 x 39 x 39 um), centred between p and the a/b midpoint,
plus two marker channels (Gaussian at p; Gaussians at a and b) built on the GPU from the relative coordinates."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STRIDE = np.array([1., 2., 2.], np.float32)
CROP = (16, 48, 48)
DT = (-2, -1, 0, 1, 2)


def load_movie(path, blosc_single=True):
    if blosc_single:
        import numcodecs.blosc; numcodecs.blosc.use_threads = False
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
    """rows: list of (t, p_zyx, a_zyx, b_zyx) in full-resolution voxel coords. Returns X [R,5,16,48,48] fp16, rel [R,3,3] fp32."""
    R = len(rows); X = np.empty((R, len(DT)) + CROP, np.float16); rel = np.empty((R, 3, 3), np.float32)
    half = np.array(CROP) // 2
    byt = {}
    for i, r in enumerate(rows): byt.setdefault(int(r[0]), []).append(i)
    for t in sorted(byt):
        fr = [fc.get(t + d) for d in DT]; shp = fr[0].shape
        for i in byt[t]:
            _, p, a, b = rows[i]
            ps = np.asarray(p, np.float32) / STRIDE; as_ = np.asarray(a, np.float32) / STRIDE; bs = np.asarray(b, np.float32) / STRIDE
            c = np.rint((ps + (as_ + bs) / 2) / 2).astype(int); lo = c - half
            sl = [np.clip(np.arange(l, l + n), 0, s - 1) for l, n, s in zip(lo, CROP, shp)]
            ix = np.ix_(*sl)
            for k in range(len(DT)): X[i, k] = fr[k][ix]
            rel[i] = np.stack([ps - lo, as_ - lo, bs - lo])
    return X, rel


def markers(rel, shape=CROP, sig=(1.2, 2.5, 2.5)):
    """rel [B,3,3] (p,a,b) -> [B,2,Z,Y,X] (p marker, a+b marker) on rel.device."""
    dev = rel.device; B = rel.shape[0]
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
        # x [B,5,Z,Y,X] fp16 intensities (0..3); rel [B,3,3]
        x = x.float()
        if self.nframes != x.shape[1]:
            o = (x.shape[1] - self.nframes) // 2; x = x[:, o:o + self.nframes]
        m = x.mean((1, 2, 3, 4), keepdim=True); s = x.std((1, 2, 3, 4), keepdim=True).clamp_min(.03)
        x = ((x - m) / s).clamp(-5, 10)
        h = self.body(torch.cat([x, 2 * markers(rel, tuple(x.shape[-3:]))], 1))
        h = torch.cat([h.mean((2, 3, 4)), h.amax((2, 3, 4))], 1)
        return self.head(h).squeeze(-1)
