"""Score pool rows of every movie with a b1-style checkpoint (artifact CellEventNet), identical rows as DivNet.
usage: score_b1.py <ckpt> <outname>"""
import os, sys, glob
sys.path.insert(0, '/workspace/art_b56/artifact_bundle')
import numpy as np, torch
from cell_event import load_event_model
D = '/dev/shm/divnet'
ck, outn = sys.argv[1], sys.argv[2]
model, _ = load_event_model(ck, 'cuda'); model.eval(); res = {}
with torch.inference_mode():
    for f in sorted(glob.glob(D + '/*.npz')):
        n = os.path.basename(f)[:-4]; d = np.load(f); T = d['B1T']; s = np.full(len(T), np.nan, np.float32); m = T[:, 0] >= 0
        if m.any():
            X = d['B1X']; emb = []
            for i in range(0, len(X), 384):
                with torch.autocast('cuda', dtype=torch.float16):
                    emb.append(model.encode(torch.from_numpy(X[i:i + 384]).cuda().float()).float())
            E = torch.cat(emb); t = torch.from_numpy(T[m]).long().cuda(); g = torch.from_numpy(d['B1G'][m]).cuda()
            s[m] = model.fork_logits(E[t[:, 0]], E[t[:, 1]], E[t[:, 2]], g).float().sigmoid().cpu().numpy()
        res[n] = s
np.savez('/workspace/cl/p16/divnet/models/%s_pred.npz' % outn, **res); print('B1_SCORED', outn, len(res))
