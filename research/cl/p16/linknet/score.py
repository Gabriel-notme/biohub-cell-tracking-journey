"""score all rl_cands 'need' pairs of one embryo with a trained model. usage: score.py <tag> <emb>"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, '/workspace/cl/p16/linknet')
from ln_common import *
tag, EMB = sys.argv[1], sys.argv[2]
OUT = '/workspace/cl/p16/linknet/out/%s' % tag; os.makedirs(OUT + '/scores', exist_ok=True)
ck = torch.load(OUT + '/model.pt', map_location='cuda'); net = LinkNet().cuda(); net.load_state_dict(ck['sd']); net.eval()
bias = np.log(ck['prior_nat'] / (1 - ck['prior_nat'])) - np.log(ck['prior_s'] / (1 - ck['prior_s']))
scm = movie_list(EMB); st = Store(scm, 'cuda'); t0 = time.time(); n = 0
with torch.no_grad():
    for s, m in scm:
        P = st.tab[m]['need']; lg = np.zeros(len(P), np.float32)
        if len(P):
            inf = st.pair_info(m, P)
            for a in range(0, len(P), 1024):
                idx = np.arange(a, min(len(P), a + 1024)); x, g = st.batch(inf, idx)
                with torch.autocast('cuda', dtype=torch.bfloat16): lg[idx] = net(x, g).float().cpu().numpy()
        np.savez(OUT + '/scores/%s__%s.npz' % (s, m), P=P, lg=lg + bias); n += len(P)
print('scored', n, round(time.time() - t0, 1), flush=True)
