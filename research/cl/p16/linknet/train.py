"""LOEO training of LinkNet.  usage: train.py <tag> <train_emb> <steps> [fold k/n]  (fold: train on movies of train_emb with index%n!=k,
score the held-out fold movies; without fold: train on all train_emb movies, score all movies of BOTH embryos)
-> out/<tag>/model.pt, out/<tag>/scores/<movie>.npz (need pairs, logits)"""
import os, sys, time, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/cl/p16/linknet')
from ln_common import *
tag, EMB, STEPS = sys.argv[1], sys.argv[2], int(sys.argv[3])
FOLD = sys.argv[4] if len(sys.argv) > 4 else None
OUT = '/workspace/cl/p16/linknet/out/%s' % tag; os.makedirs(OUT + '/scores', exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
dev = 'cuda'
allm = movie_list()
trm = [x for x in allm if x[1].startswith(EMB)]
if FOLD:
    k, n = map(int, FOLD.split('/')); trm_all = trm
    trm = [x for i, x in enumerate(trm_all) if i % n != k]; scm = [x for i, x in enumerate(trm_all) if i % n == k]
    load = trm_all
else:
    scm = []; load = trm
t0 = time.time(); st = Store(load, dev); print('store', round(time.time() - t0, 1), 's', torch.cuda.memory_allocated() / 1e9, 'GB', flush=True)
# training table
infos, Ys, W = [], [], []
for s, m in trm:
    tb = st.tab[m]; P = tb['P']
    if len(P) == 0: continue
    inf = st.pair_info(m, P); infos.append(inf); Y = tb['Y'].astype(np.float32); Ys.append(Y)
    ise = tb['par'][P[:, 1]] == P[:, 0]; hard = (ise & (Y == 0)) | (~ise & (Y == 1))
    W.append(np.where(hard, 8., 1.))
info = {k: np.concatenate([i[k] for i in infos]) for k in infos[0]}; Y = np.concatenate(Ys); W = np.concatenate(W)
pos = Y == 1
W[pos] *= 0.35 / W[pos].sum(); W[~pos] *= 0.65 / W[~pos].sum()
prior_nat = pos.mean(); prior_s = 0.35
print('train pairs', len(Y), 'pos', int(pos.sum()), 'movies', len(trm), flush=True)
cdf = np.cumsum(W); cdf /= cdf[-1]
net = LinkNet().to(dev).to(memory_format=torch.channels_last_3d)
opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=STEPS, pct_start=0.05)
BS = 512; t0 = time.time(); la = 0.
for it in range(STEPS):
    idx = np.searchsorted(cdf, np.random.rand(BS))
    x, g = st.batch(info, idx, aug=True); y = torch.from_numpy(Y[idx]).to(dev)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        lo = net(x.contiguous(memory_format=torch.channels_last_3d), g)
    loss = F.binary_cross_entropy_with_logits(lo.float(), y)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.); opt.step(); sched.step()
    la = .98 * la + .02 * loss.item() if it else loss.item()
    if it % 250 == 0 or it == STEPS - 1: print('it %d loss %.4f  %.1fs' % (it, la, time.time() - t0), flush=True)
torch.save(dict(sd=net.state_dict(), prior_nat=float(prior_nat), prior_s=prior_s, emb=EMB, fold=FOLD), OUT + '/model.pt')
# scoring
net.eval(); t0 = time.time(); npair = 0
bias = np.log(prior_nat / (1 - prior_nat)) - np.log(prior_s / (1 - prior_s))
with torch.no_grad():
    for s, m in scm:
        tb = st.tab[m]; P = tb['need']
        lg = np.zeros(len(P), np.float32)
        if len(P):
            inf = st.pair_info(m, P)
            for a in range(0, len(P), 1024):
                idx = np.arange(a, min(len(P), a + 1024)); x, g = st.batch(inf, idx)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    lg[idx] = net(x.contiguous(memory_format=torch.channels_last_3d), g).float().cpu().numpy()
        np.savez(OUT + '/scores/%s__%s.npz' % (s, m), P=P, lg=lg + bias)
        npair += len(P)
print('scored', npair, 'pairs in', round(time.time() - t0, 1), 's', flush=True)
