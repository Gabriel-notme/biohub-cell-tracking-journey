import os, sys, json, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
torch.backends.cudnn.benchmark = True

class DivNet(nn.Module):
    def __init__(self, cin=4, w=32):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv3d(i, o, 3, padding=1, bias=False), nn.BatchNorm3d(o), nn.SiLU(), nn.Conv3d(o, o, 3, padding=1, bias=False), nn.BatchNorm3d(o), nn.SiLU())
        self.b1 = blk(cin, w); self.b2 = blk(w, 2 * w); self.b3 = blk(2 * w, 4 * w); self.pool = nn.MaxPool3d(2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(8 * w + 4 * w, 64), nn.SiLU(), nn.Linear(64, 1))
    def forward(self, x):
        x = self.b1(x); x = self.pool(x); x = self.b2(x); x = self.pool(x); x = self.b3(x)
        c = x[:, :, 1:3, 3:5, 3:5].mean((2, 3, 4))
        g = torch.cat([x.mean((2, 3, 4)), x.amax((2, 3, 4)), c], 1)
        return self.head(g).squeeze(1)

def aug(x):
    # x: [B,4,Z,Y,X] float
    if np.random.rand() < .5: x = x.flip(3)
    if np.random.rand() < .5: x = x.flip(4)
    if np.random.rand() < .5: x = x.flip(2)
    k = np.random.randint(4); x = torch.rot90(x, k, (3, 4))
    s = torch.empty(x.shape[0], 1, 1, 1, 1, device=x.device).uniform_(.75, 1.3)
    x = x * s + torch.randn_like(x) * .03
    dz, dy, dx = np.random.randint(-1, 2), np.random.randint(-2, 3), np.random.randint(-2, 3)
    return torch.roll(x, (dz, dy, dx), (2, 3, 4))

def load(p):
    return np.load(p + '_X.npy', mmap_mode='r'), np.load(p + '_y.npy')

def predict(model, X, dev, bs=512, tta=True):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = torch.from_numpy(np.asarray(X[i:i + bs], np.float32)).to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = model(x).float()
                if tta:
                    lg = lg + model(x.flip(3)).float() + model(x.flip(4)).float() + model(torch.rot90(x, 1, (3, 4))).float(); lg = lg / 4
            out.append(torch.sigmoid(lg).cpu().numpy())
    return np.concatenate(out)

if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0; outp = sys.argv[2] if len(sys.argv) > 2 else '/workspace/divnet/divnet_s0.pt'
    torch.manual_seed(seed); np.random.seed(seed)
    dev = 'cuda'
    Xtr, ytr = load('/workspace/divnet/train'); Xau, yau = load('/workspace/divnet/audit'); Xho, yho = load('/workspace/divnet/hold')
    Xtr = torch.from_numpy(np.asarray(Xtr)).to(dev)  # fp16 on GPU
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(dev)
    pos_idx = np.flatnonzero(ytr == 1); neg_idx = np.flatnonzero(ytr == 0)
    print('train', Xtr.shape, 'pos', len(pos_idx), 'audit pos', int(yau.sum()), 'hold pos', int(yho.sum()), flush=True)
    model = DivNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    steps_per_epoch = 150; epochs = 24; bs = 128; npos = 24
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps_per_epoch * epochs, pct_start=0.15)
    best = (-1, None); hist = []
    for ep in range(epochs):
        model.train(); t0 = time.time(); tl = 0
        for st in range(steps_per_epoch):
            b = np.concatenate([np.random.choice(pos_idx, npos), np.random.choice(neg_idx, bs - npos)])
            x = aug(Xtr[b].float()); y = ytr_t[b]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = model(x).float()
            loss = nn.functional.binary_cross_entropy_with_logits(lg, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        pa = predict(model, Xau, dev); ap = average_precision_score(yau, pa); auc = roc_auc_score(yau, pa)
        hist.append({'ep': ep, 'loss': tl / steps_per_epoch, 'audit_ap': ap, 'audit_auc': auc})
        print('ep', ep, 'loss %.4f audit AP %.4f AUC %.4f %.1fs' % (tl / steps_per_epoch, ap, auc, time.time() - t0), flush=True)
        if ap > best[0]: best = (ap, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    ph = predict(model, Xho, dev); pa = predict(model, Xau, dev)
    res = {'seed': seed, 'best_ep': best[2], 'audit_ap': float(average_precision_score(yau, pa)), 'hold_ap': float(average_precision_score(yho, ph)), 'hold_auc': float(roc_auc_score(yho, ph)), 'hist': hist}
    print('RESULT', json.dumps({k: v for k, v in res.items() if k != 'hist'}), flush=True)
    torch.save({'state_dict': best[1], 'config': {'cin': 4, 'w': 32}, 'result': res}, outp)
