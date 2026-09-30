"""Shared data/model code for linknet: GPU-resident movie volumes, pair crops with rendered track hypothesis."""
import os, glob, pickle, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
VOL = '/dev/shm/linknet/vol'; TAB = '/workspace/cl/p16/linknet/tab'
UM = np.array([1.625, .8125, .8125], np.float32)   # pooled voxel size (z, y, x)
CZ, CY, CX = 12, 32, 32
KT = (-2, -1, 0, 1, 2)
SIG = 1.5


def to_pooled(zyx):
    z = zyx[..., 0]; y = (zyx[..., 1] - .5) / 2; x = (zyx[..., 2] - .5) / 2
    return np.stack([z, y, x], -1).astype(np.float32)


def movie_list(emb=None):
    fs = sorted(glob.glob(TAB + '/*.pkl')); out = []
    for f in fs:
        s, m = os.path.basename(f)[:-4].split('__')
        if emb is None or m.startswith(emb): out.append((s, m))
    return out


class Store:
    """All volumes (uint8) + rendered detection maps concatenated along T on the GPU."""
    def __init__(self, movies, dev):
        self.dev = dev; self.movies = movies; self.tab = {}; self.off = {}; self.T = {}
        vols = []; o = 0
        for s, m in movies:
            tb = pickle.load(open('%s/%s__%s.pkl' % (TAB, s, m), 'rb')); self.tab[m] = tb
            v = np.load('%s/%s.npy' % (VOL, m), mmap_mode='r'); self.off[m] = o; self.T[m] = v.shape[0]; o += v.shape[0]
            vols.append(v)
        self.shape = vols[0].shape[1:]
        self.vol = torch.empty((o,) + tuple(self.shape), dtype=torch.uint8, device=dev)
        self.det = torch.empty((o,) + tuple(self.shape), dtype=torch.uint8, device=dev)
        # gaussian blur kernels for detection map (sigma 1.5um)
        def k1(sp):
            r = int(np.ceil(3 * SIG / sp)); x = np.arange(-r, r + 1) * sp; k = np.exp(-x ** 2 / (2 * SIG ** 2)); return torch.tensor(k, dtype=torch.float32, device=dev)
        kz, ky, kx = k1(UM[0]), k1(UM[1]), k1(UM[2])
        for (s, m), v in zip(movies, vols):
            a = self.off[m]; T = v.shape[0]
            self.vol[a:a + T] = torch.from_numpy(np.ascontiguousarray(v)).to(dev)
            tb = self.tab[m]; p = np.rint(to_pooled(tb['zyx'])).astype(np.int64)
            Z, Y, X = self.shape
            p[:, 0] = np.clip(p[:, 0], 0, Z - 1); p[:, 1] = np.clip(p[:, 1], 0, Y - 1); p[:, 2] = np.clip(p[:, 2], 0, X - 1)
            for t0 in range(0, T, 25):
                t1 = min(T, t0 + 25); sel = (tb['t'] >= t0) & (tb['t'] < t1)
                d = torch.zeros((t1 - t0, 1, Z, Y, X), device=dev)
                pp = torch.from_numpy(p[sel]).to(dev); tt = torch.from_numpy(tb['t'][sel].astype(np.int64) - t0).to(dev)
                d[tt, 0, pp[:, 0], pp[:, 1], pp[:, 2]] = 1.
                d = F.conv3d(d, kz.view(1, 1, -1, 1, 1), padding=(len(kz) // 2, 0, 0))
                d = F.conv3d(d, ky.view(1, 1, 1, -1, 1), padding=(0, len(ky) // 2, 0))
                d = F.conv3d(d, kx.view(1, 1, 1, 1, -1), padding=(0, 0, len(kx) // 2))
                self.det[a + t0:a + t1] = (d[:, 0].clamp(0, 1) * 255).round().to(torch.uint8)
        self.mi = {m: i for i, (s, m) in enumerate(movies)}

    def pair_info(self, m, P):
        """per pair: frame index base, crop center (pooled vox int), 5 track points (um rel. to center), valid, geometry."""
        tb = self.tab[m]; zyx = to_pooled(tb['zyx']); par = tb['par']; ch1 = tb['ch1']
        x, y = P[:, 0], P[:, 1]
        a1 = par[x]; a2 = np.where(a1 >= 0, par[np.maximum(a1, 0)], -1); d1 = ch1[y]
        pts = np.stack([a2, a1, x, y, d1], 1)  # node idx or -1
        val = pts >= 0
        um = zyx * UM
        mid = (zyx[x] + zyx[y]) / 2; cen = np.rint(mid).astype(np.int64)
        cum = cen * UM
        rel = np.where(val[..., None], um[np.maximum(pts, 0)] - cum[:, None, :], 0.).astype(np.float32)
        t = tb['t'][x].astype(np.int64)
        disp = um[y] - um[x]; dist = np.linalg.norm(disp, axis=1)
        vx = np.where(val[:, 1:2], um[x] - um[np.maximum(a1, 0)], 0.); vy = np.where(val[:, 4:5], um[np.maximum(d1, 0)] - um[y], 0.)
        nvx = np.linalg.norm(vx, axis=1); nvy = np.linalg.norm(vy, axis=1)
        geo = np.stack([dist, np.abs(disp[:, 0]), np.linalg.norm(disp[:, 1:], axis=1), nvx, nvy,
                        np.linalg.norm(disp - vx, axis=1) * val[:, 1], np.linalg.norm(disp - vy, axis=1) * val[:, 4],
                        (disp * vx).sum(1) / (dist * nvx + 1e-3), (disp * vy).sum(1) / (dist * nvy + 1e-3),
                        val[:, 1].astype(float), val[:, 4].astype(float), val[:, 0].astype(float)], 1).astype(np.float32)
        geo[:, :7] /= 10.
        return dict(mov=np.full(len(P), self.mi[m], np.int64), t=t, cen=cen, rel=rel, val=val, geo=geo)

    def batch(self, info, idx, aug=False):
        dev = self.dev
        mov = info['mov'][idx]; t = info['t'][idx]; cen = info['cen'][idx]
        B = len(idx)
        offs = np.array([self.off[self.movies[i][1]] for i in mov]); Ts = np.array([self.T[self.movies[i][1]] for i in mov])
        fr = offs[:, None] + np.clip(t[:, None] + np.array(KT)[None], 0, Ts[:, None] - 1)  # B,5
        Z, Y, X = self.shape
        zi = np.clip(cen[:, 0:1] + np.arange(CZ)[None] - CZ // 2, 0, Z - 1)
        yi = np.clip(cen[:, 1:2] + np.arange(CY)[None] - CY // 2, 0, Y - 1)
        xi = np.clip(cen[:, 2:3] + np.arange(CX)[None] - CX // 2, 0, X - 1)
        fr = torch.from_numpy(fr).to(dev); zi = torch.from_numpy(zi).to(dev); yi = torch.from_numpy(yi).to(dev); xi = torch.from_numpy(xi).to(dev)
        I = (fr[:, :, None, None, None], zi[:, None, :, None, None], yi[:, None, None, :, None], xi[:, None, None, None, :])
        img = self.vol[I].float() / 85.; det = self.det[I].float() / 255.
        # rendered hypothesis
        gz = (torch.arange(CZ, device=dev) - CZ // 2).float() * UM[0]; gy = (torch.arange(CY, device=dev) - CY // 2).float() * UM[1]; gx = (torch.arange(CX, device=dev) - CX // 2).float() * UM[2]
        rel = torch.from_numpy(info['rel'][idx]).to(dev); val = torch.from_numpy(info['val'][idx]).to(dev).float()
        dz = (gz[None, None, :] - rel[:, :, 0:1]) ** 2; dy = (gy[None, None, :] - rel[:, :, 1:2]) ** 2; dx = (gx[None, None, :] - rel[:, :, 2:3]) ** 2
        hyp = torch.exp(-(dz[:, :, :, None, None] + dy[:, :, None, :, None] + dx[:, :, None, None, :]) / (2 * SIG ** 2)) * val[:, :, None, None, None]
        x = torch.cat([img, hyp, det], 1)
        g = torch.from_numpy(info['geo'][idx]).to(dev)
        if aug:
            if np.random.rand() < .5: x = x.flip(-1)
            if np.random.rand() < .5: x = x.flip(-2)
            if np.random.rand() < .5: x = x.flip(-3)
            if np.random.rand() < .5: x = x.transpose(-1, -2)
            x = x * (1 + .1 * torch.randn(B, 1, 1, 1, 1, device=dev))  # mild intensity jitter on all channels is fine
        return x, g


class Res(nn.Module):
    def __init__(self, c):
        super().__init__(); self.n = nn.Sequential(nn.Conv3d(c, c, 3, padding=1, bias=False), nn.GroupNorm(8, c), nn.SiLU(), nn.Conv3d(c, c, 3, padding=1, bias=False), nn.GroupNorm(8, c))
    def forward(self, x): return F.silu(x + self.n(x))


def blk(a, b, st): return nn.Sequential(nn.Conv3d(a, b, 3, stride=st, padding=1, bias=False), nn.GroupNorm(8, b), nn.SiLU())


class LinkNet(nn.Module):
    def __init__(self, cin=15, w=32, ng=12):
        super().__init__()
        self.enc = nn.Sequential(blk(cin, w, 1), Res(w), blk(w, 2 * w, (1, 2, 2)), Res(2 * w), blk(2 * w, 4 * w, 2), Res(4 * w), blk(4 * w, 4 * w, 2))
        self.g = nn.Sequential(nn.Linear(ng, 64), nn.SiLU(), nn.Linear(64, 64))
        self.head = nn.Sequential(nn.Linear(8 * w + 64, 256), nn.SiLU(), nn.Dropout(.2), nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 1))
    def forward(self, x, g):
        h = self.enc(x); h = torch.cat([h.mean((-3, -2, -1)), h.amax((-3, -2, -1))], 1)
        return self.head(torch.cat([h, self.g(g)], 1))[:, 0]
