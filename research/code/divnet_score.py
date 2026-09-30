import os, sys, json, glob, time
import numpy as np, torch
sys.path.insert(0, '/workspace/code')
from divnet_data import Vol, FR, CZ, CY
from divnet_train import DivNet
# usage: divnet_score.py <graph_dir> <cands_dir> <models comma> <out_dir> <gpu>
gdir, cdir, models, odir, gpu = sys.argv[1:6]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
dev = 'cuda'
nets = []
for m in models.split(','):
    ck = torch.load(m, map_location='cpu'); net = DivNet(**ck['config']); net.load_state_dict(ck['state_dict']); net.to(dev).eval(); nets.append(net)
os.makedirs(odir, exist_ok=True)
names = sorted(os.path.basename(p).split('.')[0] for p in glob.glob(cdir + '/*.cands.json'))
shard = int(sys.argv[6]) if len(sys.argv) > 6 else 0; nsh = int(sys.argv[7]) if len(sys.argv) > 7 else 1
for name in names[shard::nsh]:
    op = odir + '/' + name + '.divnet.json'
    if os.path.exists(op): continue
    t0 = time.time()
    d = json.load(open(gdir + '/' + name + '.json')); nodes = d['nodes']
    cands = json.load(open(cdir + '/' + name + '.cands.json'))
    ps = sorted({c['p'] for c in cands})
    ps.sort(key=lambda p: nodes[str(p)]['t'])
    V = Vol(name)
    probs = {}
    for i in range(0, len(ps), 1024):
        chunk = ps[i:i + 1024]
        X = np.stack([V.crop(int(nodes[str(p)]['t']), nodes[str(p)]['z'], nodes[str(p)]['y'], nodes[str(p)]['x']) for p in chunk])
        x = torch.from_numpy(X.astype(np.float32)).to(dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            lg = sum(n(x).float() + n(x.flip(3)).float() + n(x.flip(4)).float() + n(torch.rot90(x, 1, (3, 4))).float() for n in nets) / (4 * len(nets))
        pr = torch.sigmoid(lg).cpu().numpy()
        probs.update({int(p): float(v) for p, v in zip(chunk, pr)})
    json.dump(probs, open(op, 'w'))
    print('DIVNET', name, len(ps), round(time.time() - t0, 1), flush=True)
