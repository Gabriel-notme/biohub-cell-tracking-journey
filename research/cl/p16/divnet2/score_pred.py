"""Score the other embryo's v1 rows with one saved LOEO checkpoint (for a seed whose pred file is missing). usage: score_pred.py <ckpt>"""
import sys, glob, os, numpy as np, torch
sys.path.insert(0, '/workspace/cl/p16/divnet')
import divnet_lib as L
from train_divnet import score_all
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
m = L.DivNet(**ck['config']); m.load_state_dict(ck['model']); m = m.cuda()
names = sorted(os.path.basename(f)[:-4] for f in glob.glob('/dev/shm/divnet/*.npz'))
pr = score_all(m, [n for n in names if not n.startswith(ck['train_emb'])], 'cuda')
np.savez(sys.argv[1][:-3] + '_pred.npz', **pr); print('PRED_SAVED', sys.argv[1])
