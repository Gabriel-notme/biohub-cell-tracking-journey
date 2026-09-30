"""Train DivNet models on one embryo (LOEO) or all; score every row of every movie; save weights + predictions.
usage: train_divnet.py <train_emb: 44b6|6bba|all> <tag> '<json list of {arm, seed, steps, width, nframes, lr}>'
arm: full (GT rows + pool rows) | gt (GT rows only) | pool (pool rows only). Final step checkpoint is used (no selection)."""
import os, sys, json, glob, time, math
sys.path.insert(0, '/workspace/cl/p16/divnet')
import numpy as np, torch
from torch.nn import functional as F
import divnet_lib as L
D = '/dev/shm/divnet'; OUTD = '/workspace/cl/p16/divnet/models'


def load(names, keep):
    X, R, Y = [], [], []
    for n in names:
        d = np.load(D + '/' + n + '.npz')
        m = keep(d)
        if m.sum() == 0: continue
        X.append(d['X'][m]); R.append(d['rel'][m]); Y.append(d['y'][m])
    return np.concatenate(X), np.concatenate(R), np.concatenate(Y).astype(np.float32)


def augment(x, rel):
    B = x.shape[0]; sz = torch.tensor(L.CROP, device=x.device, dtype=torch.float32)
    for ax in range(3):
        f = torch.rand(B, device=x.device) < .5
        if f.any():
            x[f] = x[f].flip(-3 + ax); rel[f, :, ax] = sz[ax] - 1 - rel[f, :, ax]
    f = torch.rand(B, device=x.device) < .5
    if f.any():
        x[f] = x[f].transpose(-1, -2); r = rel[f]; rel[f] = r[:, :, [0, 2, 1]]
    # detector-like displacement of each marker (voxels on the strided grid: z 1.6um, yx 0.8um)
    rel = rel + torch.randn_like(rel) * torch.tensor([.6, 1.2, 1.2], device=x.device)
    x = x * torch.empty(B, 1, 1, 1, 1, device=x.device).uniform_(.7, 1.4)
    x = x + torch.randn_like(x) * float(np.random.uniform(0, .04))
    return x, rel


def train_one(Xg, Rg, Yg, cfg, dev):
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    model = L.DivNet(nframes=cfg.get('nframes', 5), width=cfg.get('width', 32)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', 1e-3), weight_decay=.02)
    pos = torch.nonzero(Yg > .5).squeeze(1); neg = torch.nonzero(Yg < .5).squeeze(1)
    steps = cfg.get('steps', 2000); bs = cfg.get('batch', 128); npos = bs // 3; t0 = time.time()
    for step in range(1, steps + 1):
        lr = cfg.get('lr', 1e-3) * min(1, step / 100) * (.05 + .95 * (1 + math.cos(math.pi * step / steps)) / 2)
        for g in opt.param_groups: g['lr'] = lr
        ix = torch.cat([pos[torch.randint(len(pos), (npos,), device=dev)], neg[torch.randint(len(neg), (bs - npos,), device=dev)]])
        x = Xg[ix].float(); rel = Rg[ix].clone(); y = Yg[ix]
        x, rel = augment(x, rel)
        model.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lo = model(x, rel)
        loss = F.binary_cross_entropy_with_logits(lo.float(), y * .98 + .01)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.); opt.step()
        if step % 250 == 0: print('STEP', cfg['name'], step, round(float(loss), 4), round(time.time() - t0), flush=True)
    return model


@torch.inference_mode()
def score_all(model, names, dev):
    model.eval(); out = {}
    for n in names:
        d = np.load(D + '/' + n + '.npz'); X, R = d['X'], d['rel']; s = []
        for i in range(0, len(X), 512):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                s.append(model(torch.from_numpy(X[i:i + 512]).to(dev), torch.from_numpy(R[i:i + 512]).to(dev)).float().sigmoid().cpu().numpy())
        out[n] = np.concatenate(s) if s else np.zeros(0, np.float32)
    return out


if __name__ == '__main__':
    emb, tag, cfgs = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    dev = 'cuda'; torch.backends.cudnn.benchmark = True
    allnames = sorted(os.path.basename(f)[:-4] for f in glob.glob(D + '/*.npz'))
    tr = [n for n in allnames if emb == 'all' or n.startswith(emb)]
    os.makedirs(OUTD, exist_ok=True)
    cache = {}
    for cfg in cfgs:
        arm = cfg['arm']; cfg['name'] = '%s_%s_tr%s_s%d' % (tag, arm, emb, cfg['seed'])
        if arm not in cache:
            cache.clear(); torch.cuda.empty_cache()
            if arm == 'full': keep = lambda d: d['y'] >= 0
            elif arm == 'gt': keep = lambda d: (d['y'] >= 0) & (d['src'] == 'gt')
            elif arm == 'pool': keep = lambda d: (d['y'] >= 0) & (d['src'] != 'gt')
            X, R, Y = load(tr, keep)
            print('DATA', arm, emb, X.shape, int(Y.sum()), flush=True)
            cache[arm] = (torch.from_numpy(X).to(dev), torch.from_numpy(R).to(dev), torch.from_numpy(Y).to(dev)); X = None
        model = train_one(*cache[arm], cfg, dev)
        torch.save({'config': {'nframes': cfg.get('nframes', 5), 'width': cfg.get('width', 32)}, 'model': model.state_dict(), 'train_emb': emb, 'cfg': cfg,
                    'train_movies': tr}, OUTD + '/' + cfg['name'] + '.pt')
        if emb != 'all':
            pr = score_all(model, [n for n in allnames if not n.startswith(emb)], dev)
            np.savez(OUTD + '/' + cfg['name'] + '_pred.npz', **pr)
        print('SAVED', cfg['name'], flush=True)
    print('TRAIN_FIN', tag, emb, flush=True)
