"""Train DivNet v2 variants (see NOTES.md pre-registration). usage: train_v2.py <train_emb 44b6|6bba|all> <variant V2H|V2P|V2PH> <seeds e.g. 0,1,2>
Rows: v1 'full' rows of the training embryo (/dev/shm/divnet, y>=0) + new rows (/dev/shm/divnet2: ps / pr / hn).
Final checkpoint (no selection). For LOEO models the v1 rows of the other embryo are scored (candidate-level metrics)."""
import os, sys, json, glob, time, math
sys.path.insert(0, '/workspace/cl/p16/divnet')
import numpy as np, torch
from torch.nn import functional as F
import divnet_lib as L
sys.path.insert(0, '/workspace/cl/p16/divnet'); from train_divnet import augment, score_all
D1 = '/dev/shm/divnet'; D2 = '/dev/shm/divnet2'; OUTD = os.environ.get('OUTD', '/workspace/cl/p16/divnet2/models')


def load(names, use_hn, use_ps):
    X, R, T, K = [], [], [], []   # K: 0 real pos, 1 v1 neg, 2 ps, 3 pr, 4 hn
    for n in names:
        d = np.load(D1 + '/' + n + '.npz'); m = d['y'] >= 0
        if m.any():
            y = d['y'][m].astype(np.float32); X.append(d['X'][m]); R.append(d['rel'][m]); T.append(y * .98 + .01); K.append(np.where(y > .5, 0, 1))
        d = np.load(D2 + '/' + n + '.npz'); kd = d['kind']
        sel = np.zeros(len(kd), bool)
        if use_ps: sel |= (kd == 'ps') | (kd == 'pr')
        if use_hn: sel |= kd == 'hn'
        if sel.any():
            kd = kd[sel]; X.append(d['X'][sel]); R.append(d['rel'][sel])
            T.append(np.where(kd == 'ps', .8, .01).astype(np.float32)); K.append(np.select([kd == 'ps', kd == 'pr', kd == 'hn'], [2, 3, 4]))
    return np.concatenate(X), np.concatenate(R), np.concatenate(T).astype(np.float32), np.concatenate(K).astype(np.int64)


def run_stage(model, Xg, Rg, Tg, pools, steps, lr0, warm, dev, name, seed_off):
    """pools: list of (index tensor, n per batch)"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr0, weight_decay=.02); t0 = time.time()
    for step in range(1, steps + 1):
        lr = lr0 * min(1, step / warm) * (.05 + .95 * (1 + math.cos(math.pi * step / steps)) / 2)
        for g in opt.param_groups: g['lr'] = lr
        ix = torch.cat([ix_[torch.randint(len(ix_), (k,), device=dev)] for ix_, k in pools if k > 0])
        x = Xg[ix].float(); rel = Rg[ix].clone(); y = Tg[ix]
        x, rel = augment(x, rel)
        model.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lo = model(x, rel)
        loss = F.binary_cross_entropy_with_logits(lo.float(), y)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.); opt.step()
        if step % 500 == 0: print('STEP', name, seed_off, step, round(float(loss.detach()), 4), round(time.time() - t0), flush=True)


def train_one(Xg, Rg, Tg, Kg, variant, seed, dev, name):
    torch.manual_seed(seed); np.random.seed(seed)
    model = L.DivNet(nframes=5, width=32).to(dev)
    idx = lambda *ks: torch.nonzero(torch.isin(Kg, torch.tensor(ks, device=dev))).squeeze(1)
    pos, neg1, ps, pr, hn = idx(0), idx(1), idx(2), idx(3), idx(4)
    bs, npos = 128, 128 // 3; nneg = bs - npos; nhn = nneg // 4
    if variant == 'V2H':
        run_stage(model, Xg, Rg, Tg, [(pos, npos), (hn, nhn), (neg1, nneg - nhn)], 2000, 1e-3, 100, dev, name, 'H')
    else:
        use_hn = variant == 'V2PH'
        negu = torch.cat([neg1, pr])
        s1 = [(pos, npos - npos // 2), (ps, npos // 2)] + ([(hn, nhn), (negu, nneg - nhn)] if use_hn else [(negu, nneg)])
        run_stage(model, Xg, Rg, Tg, s1, 2000, 1e-3, 100, dev, name, 'pre')
        s2 = [(pos, npos)] + ([(hn, nhn), (neg1, nneg - nhn)] if use_hn else [(neg1, nneg)])
        run_stage(model, Xg, Rg, Tg, s2, 1000, 3e-4, 50, dev, name, 'ft')
    return model


if __name__ == '__main__':
    emb, variant, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(',')]
    steps_override = os.environ.get('SMOKE')
    dev = 'cuda'; torch.backends.cudnn.benchmark = True
    allnames = sorted(os.path.basename(f)[:-4] for f in glob.glob(D1 + '/*.npz'))
    tr = [n for n in allnames if emb == 'all' or n.startswith(emb)]
    use_hn = variant in ('V2H', 'V2PH'); use_ps = variant in ('V2P', 'V2PH')
    X, R, T, K = load(tr, use_hn, use_ps)
    print('DATA', variant, emb, X.shape, {k: int((K == k).sum()) for k in range(5)}, flush=True)
    Xg, Rg, Tg, Kg = (torch.from_numpy(a).to(dev) for a in (X, R, T, K)); X = None
    os.makedirs(OUTD, exist_ok=True)
    for seed in seeds:
        name = '%s_tr%s_s%d' % (variant, emb, seed)
        if steps_override:
            _rs = run_stage
            def run_stage(*a, **k):
                a = list(a); a[5] = int(steps_override); return _rs(*a, **k)
            globals()['run_stage'] = run_stage
        model = train_one(Xg, Rg, Tg, Kg, variant, seed, dev, name)
        torch.save({'config': {'nframes': 5, 'width': 32}, 'model': model.state_dict(), 'train_emb': emb, 'variant': variant, 'seed': seed,
                    'train_movies': tr}, OUTD + '/' + name + '.pt')
        if emb != 'all':
            pr = score_all(model, [n for n in allnames if not n.startswith(emb)], dev)
            np.savez(OUTD + '/' + name + '_pred.npz', **pr)
        print('SAVED', name, flush=True)
    print('TRAIN_FIN', variant, emb, flush=True)
