"""Leave-one-embryo-out fine-tuning of the b1 event model's fork head (+ encoder) on division-completion candidates.
usage: ft_train.py <gpu> <train_embryo> <test_embryo> <steps>
Reports weighted AUC and precision-at-threshold on the test embryo for: base b1 (original parent), zero-shot b1 on the
virtual-parent inputs, and the fine-tuned model. Negatives are re-weighted to the full candidate population."""
import os, sys, glob, json
gpu, src, dst, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
ART = '/workspace/art_b56/artifact_bundle'; sys.path.insert(0, ART)
import numpy as np
import torch
from cell_event import load_event_model


def load(emb):
    Xs, Gs, ys, bs, ws, ts, us = [], [], [], [], [], [], []
    for f in sorted(glob.glob('/workspace/cl/ft/*__%s_*.npz' % emb)):
        z = np.load(f)
        if int(z['n']) == 0: continue
        s = f.split('/')[-1].split('__')[0]
        y = z['y']; w = np.ones(len(y), np.float32)
        neg = np.flatnonzero(y == 0); nt = int(z['n_neg_total']); nh = min(40, nt)
        if len(neg) > nh:
            w[neg[nh:]] = (nt - nh) / (len(neg) - nh)
        Xs.append(z['X']); Gs.append(z['G']); ys.append(y); bs.append(z['base']); ws.append(w); ts.append(z['typ']); us.append(np.full(len(y), s in ('hold36', 'prev4')))
    return (np.concatenate(Xs), np.concatenate(Gs), np.concatenate(ys), np.concatenate(bs), np.concatenate(ws), np.concatenate(ts), np.concatenate(us))


def tf(x, i):
    if i == 0: return x
    if i <= 3: return x.flip([(-1,), (-2,), (-2, -1)][i - 1])
    if i <= 5: return torch.rot90(x, [1, 3][i - 4], (-2, -1))
    if i == 6: return x.transpose(-1, -2)
    return torch.rot90(x, 1, (-2, -1)).transpose(-1, -2)


def logits(model, X, G, aug=0):
    n = X.shape[0]; x = X.reshape(n * 3, *X.shape[2:])
    x = tf(x, aug)
    with torch.autocast('cuda', dtype=torch.float16):
        e = model.encode(x).float().view(n, 3, -1)
    return model.fork_logits(e[:, 0], e[:, 1], e[:, 2], G).float()


def predict(model, X, G, tta=True):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256]).cuda().float(); gb = torch.from_numpy(G[i:i + 256]).cuda()
            L = [logits(model, xb, gb, a) for a in (range(8) if tta else [0])]
            out.append(torch.stack(L).mean(0).cpu().numpy())
    return 1 / (1 + np.exp(-np.concatenate(out)))


def wauc(p, y, w):
    pos = p[y == 1]; neg = p[y == 0]; wn = w[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    o = np.argsort(neg); ns = neg[o]; cw = np.cumsum(wn[o])
    idx = np.searchsorted(ns, pos, side='left'); below = np.where(idx > 0, cw[np.maximum(idx - 1, 0)], 0.0)
    return float(below.sum() / (len(pos) * wn.sum()))


def report(tag, p, y, w, typ, uns):
    line = '%-22s' % tag
    for nm, m in [('all', np.ones(len(y), bool)), ('stolen', typ == 1), ('start', typ == 0), ('unseen', uns)]:
        line += ' | %s AUC %.3f' % (nm, wauc(p[m], y[m], w[m]))
    for th in [0.5, 0.8, 0.9, 0.97]:
        k = p >= th; tp = int(y[k].sum()); fp = float(w[k & (y == 0)].sum())
        line += ' | >=%.2f P %d N~%.0f' % (th, tp, fp)
    print(line, flush=True)


torch.manual_seed(0); np.random.seed(0)
Xa, Ga, ya, ba, wa, ta, ua = load(src); Xb, Gb, yb, bb, wb, tb, ub = load(dst)
print('train %s: pos %d neg %d | test %s: pos %d neg %d (weighted neg %.0f)' % (src, ya.sum(), (1 - ya).sum(), dst, yb.sum(), (1 - yb).sum(), wb[yb == 0].sum()), flush=True)
report('base b1 (orig parent)', np.clip(bb, 0, 1), yb, wb, tb, ub)
model, _ = load_event_model(os.path.join(ART, 'b1_best.pt'), 'cuda')
report('b1 zero-shot vparent', predict(model, Xb, Gb, tta=False), yb, wb, tb, ub)
report('b1 zero-shot vp+TTA', predict(model, Xb, Gb, tta=True), yb, wb, tb, ub)
opt = torch.optim.AdamW([{'params': model.fork.parameters(), 'lr': 3e-4}, {'params': [p for n, p in model.named_parameters() if not n.startswith('fork.')], 'lr': 3e-5}], weight_decay=1e-4)
Gt = torch.from_numpy(Ga).cuda(); pos_i = np.flatnonzero(ya == 1); neg_i = np.flatnonzero(ya == 0)
for step in range(steps):
    model.train()
    bi = np.concatenate([np.random.choice(pos_i, 16), np.random.choice(neg_i, 48)])
    xb = torch.from_numpy(Xa[bi]).cuda().float(); yb_t = torch.from_numpy(ya[bi].astype(np.float32)).cuda()
    lg = logits(model, xb, Gt[bi], aug=int(np.random.randint(8)))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(lg, yb_t)
    opt.zero_grad(); loss.backward(); opt.step()
    if (step + 1) % max(1, steps // 4) == 0:
        report('fine-tuned step %d' % (step + 1), predict(model, Xb, Gb, tta=True), yb, wb, tb, ub)
