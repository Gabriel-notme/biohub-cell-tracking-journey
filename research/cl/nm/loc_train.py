"""LOEO learnability test of a dedicated offset regressor (patch at pred -> GT - pred, range +-12 um).
usage: loc_train.py <train_emb> <init: b5|scratch> <tail_weight> <minutes>
Encoder = B5 CellEventNet (init from centroid_v1_frozen = in-sample for both embryos -> optimistic), new head 12*tanh.
Reports on the held-out embryo per typ: cos(pred, true), frac full-apply within 7 um, frac pushed out (typ0/1), and a gated variant
(apply only if predicted |delta| >= 2 um)."""
import os, sys, json, glob, time, math
os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, '/workspace/art_b56/artifact_bundle')
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
from cell_event import CellEventNet
tr_emb, init, tw, minutes = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
te_emb = '6bba' if tr_emb == '44b6' else '44b6'
torch.manual_seed(0); np.random.seed(0)


def load(emb):
    X, Y, Fr, T, D = [], [], [], [], []
    for f in sorted(glob.glob('/workspace/cl/nm/loc_ds/%s_*.npz' % emb)):
        d = np.load(f); X.append(d['x']); Y.append(d['y']); Fr.append(d['f']); T.append(d['typ']); D.append(d['d0'])
    return np.concatenate(X), np.concatenate(Y), np.concatenate(Fr), np.concatenate(T), np.concatenate(D)


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.base = CellEventNet(**cfg); self.head = nn.Sequential(nn.Linear(67, 128), nn.SiLU(), nn.Linear(128, 3))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, f):
        z = self.base.encode(x); return 12 * torch.tanh(self.head(torch.cat([z, f.to(z.dtype)], 1)) / 12)


ck = torch.load('/workspace/art_b56/artifact_bundle/centroid_v1_frozen.pt', map_location='cpu', weights_only=False)
net = Net(ck['config']['base_config'])
if init == 'b5':
    sd = {k[5:]: v for k, v in ck['model'].items() if k.startswith('base.')}; print(net.base.load_state_dict(sd))
net = net.cuda()
x, y, fr, ty, d0 = load(tr_emb)
print('train', tr_emb, len(x), np.bincount(ty), flush=True)
xg = torch.from_numpy(x).cuda(); yg = torch.from_numpy(y).cuda(); fg = torch.from_numpy(fr).cuda()
w = np.where(ty > 0, tw, 1.0); w = w / w.sum(); wg = torch.from_numpy(w).cuda()
opt = torch.optim.AdamW([{'params': net.base.parameters(), 'lr': 5e-5 if init == 'b5' else 5e-4}, {'params': net.head.parameters(), 'lr': 5e-4}], weight_decay=.02)
spacing = torch.tensor([1.625, .8125, .8125], device='cuda')
start = time.time(); step = 0; B = 256
while time.time() - start < minutes * 60:
    step += 1
    ix = torch.multinomial(wg, B, replacement=True)
    xx = xg[ix].float(); yy = yg[ix].clone(); ff = fg[ix].clone()
    for a in range(3):
        if np.random.rand() < .5: xx = xx.flip(a + 2); yy[:, a] *= -1; ff[:, a] = -ff[:, a] - spacing[a]
    if np.random.rand() < .5: xx = xx.transpose(-1, -2); yy = yy[:, [0, 2, 1]]; ff = ff[:, [0, 2, 1]]
    xx = xx * (.6 + .8 * torch.rand((B, 1, 1, 1, 1), device='cuda'))
    frac_t = min(1., (time.time() - start) / (minutes * 60))
    for g, base_lr in zip(opt.param_groups, [5e-5 if init == 'b5' else 5e-4, 5e-4]): g['lr'] = base_lr * (.05 + .95 * .5 * (1 + math.cos(math.pi * frac_t))) * min(1., step / 100)
    net.train(); opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16): p = net(xx, ff)
    loss = F.smooth_l1_loss(p.float(), yy, beta=1.); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.); opt.step()
    if step % 500 == 0: print('step', step, float(loss), round(time.time() - start), flush=True)
del xg, yg, fg
x, y, fr, ty, d0 = load(te_emb)
net.eval(); P = []
with torch.inference_mode():
    for a in range(0, len(x), 512):
        with torch.autocast('cuda', dtype=torch.bfloat16): P.append(net(torch.from_numpy(x[a:a + 512]).cuda().float(), torch.from_numpy(fr[a:a + 512]).cuda()).float().cpu().numpy())
P = np.concatenate(P)
dn = np.linalg.norm(y - P, axis=1); dp = np.linalg.norm(P, axis=1); d_now = np.linalg.norm(y, axis=1)
cos = (P * y).sum(1) / (dp * d_now + 1e-9)
res = {'train': tr_emb, 'test': te_emb, 'init': init, 'tw': tw, 'steps': step}
for k, nm in enumerate(['M', 'M57', 'U']):
    m = ty == k
    r = dict(n=int(m.sum()), cos_med=float(np.median(cos[m])), dlen_med=float(np.median(dp[m])), d_before=float(np.median(d_now[m])), d_after=float(np.median(dn[m])),
             within7_before=float((d_now[m] <= 7).mean()), within7_after=float((dn[m] <= 7).mean()), improved_1um=float((dn[m] < d_now[m] - 1).mean()),
             worse_1um=float((dn[m] > d_now[m] + 1).mean()))
    for gate in [2., 3., 4.]:
        app = dp >= gate; dd = np.where(app, dn, d_now)
        r['gate%.0f_applied' % gate] = float(app[m].mean()); r['gate%.0f_within7' % gate] = float((dd[m] <= 7).mean())
    res[nm] = r
print(json.dumps(res, indent=1))
json.dump(res, open('/workspace/cl/nm/loc_train_%s_%s_%g.json' % (tr_emb, init, tw), 'w'))
np.save('/workspace/cl/nm/loc_pred_%s_%s_%g.npy' % (te_emb, init, tw), P)
