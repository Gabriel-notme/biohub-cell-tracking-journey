"""Does D4 (yx-plane) test-time augmentation of the b1 fork head separate true division candidates better?
Joins P-stage candidate dumps (fork prob on the B5 graph) with GT labels from cand_lab (cl2/<set>/<movie>.json).
For candidates with base prob >= LO, recompute fork prob as mean logit over 8 D4 transforms."""
import os, sys, json, glob
os.environ['BIOHUB_ART'] = '/workspace/art_b56/artifact_bundle'
sys.path.insert(0, '/workspace/art_b56/artifact_bundle'); sys.path.insert(0, '/workspace/p12ds')
from pathlib import Path
import numpy as np
import torch
from cell_event import SCALE, Movie, chain, fork_geometry, load_event_model
LO = float(os.environ.get('TTA_LO', '0.3'))


def tf(x, i):
    if i == 0: return x
    if i <= 3: return x.flip([(-1,), (-2,), (-2, -1)][i - 1])
    if i <= 5: return torch.rot90(x, [1, 3][i - 4], (-2, -1))
    if i == 6: return x.transpose(-1, -2)
    return torch.rot90(x, 1, (-2, -1)).transpose(-1, -2)


def run(tag, dump_dir, graph_dir, data_dir, gpu):
    torch.cuda.set_device(gpu)
    model, _ = load_event_model('/workspace/art_b56/artifact_bundle/b1_best.pt', 'cuda:%d' % gpu)
    import div_complete as dc
    out = []
    for f in sorted(glob.glob(dump_dir + '/*.json')):
        name = Path(f).stem
        lf = Path('/workspace/cl/cl2/%s/%s.json' % (tag, name))
        if not lf.exists(): continue
        cands = json.load(open(f)); labs = {(r['p'], r['a'], r['b']): r['lab'] for r in json.load(open(lf))}
        sel = [c for c in cands if c['fork'] >= LO]
        if not sel: continue
        g = json.load(open(Path(graph_dir) / (name + '.json')))
        nodes = {int(k): v for k, v in g['nodes'].items()}
        out_, prev, pos = dc.structure(nodes, g['edges'])[0], dc.structure(nodes, g['edges'])[1], {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
        need = sorted({x for c in sel for x in (c['p'], c['a'], c['b'])}, key=lambda n: (nodes[n]['t'], n))
        lk = {n: i for i, n in enumerate(need)}
        mv = Movie(Path(data_dir) / (name + '.zarr'), context=5)
        X = mv.patches([nodes[n]['t'] for n in need], [[nodes[n][k] for k in 'zyx'] for n in need])
        geo = np.stack([fork_geometry(chain(c['p'], prev, pos), chain(c['a'], out_, pos), chain(c['b'], out_, pos)) for c in sel]).astype(np.float32)
        ix = np.array([[lk[c['p']], lk[c['a']], lk[c['b']]] for c in sel])
        logits = []
        with torch.inference_mode():
            xt = torch.from_numpy(X).to('cuda:%d' % gpu, dtype=torch.float32)
            gg = torch.from_numpy(geo).to('cuda:%d' % gpu)
            for i in range(8):
                embs = []
                for s in range(0, len(xt), 256):
                    with torch.autocast('cuda', dtype=torch.float16):
                        embs.append(model.encode(tf(xt[s:s + 256], i)).float())
                E = torch.cat(embs)
                lg = model.fork_logits(E[ix[:, 0]], E[ix[:, 1]], E[ix[:, 2]], gg).float().cpu().numpy()
                logits.append(lg)
        L = np.stack(logits)
        for c, l0, lm in zip(sel, L[0], L.mean(0)):
            out.append(dict(movie=name, typ=c['typ'], lab=labs.get((c['p'], c['a'], c['b']), 'X'), base=c['fork'], p0=float(1 / (1 + np.exp(-l0))), tta=float(1 / (1 + np.exp(-lm))),
                            sd=float(L[:, len(out) - len(out)].std()) if False else 0.))
    return out


if __name__ == '__main__':
    tag, dump_dir, graph_dir, data_dir, gpu = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    R = run(tag, dump_dir, graph_dir, data_dir, gpu)
    json.dump(R, open('/workspace/cl/tta_%s.json' % tag, 'w'))
    from collections import Counter
    print(tag, 'rows', len(R), Counter((r['typ'], r['lab']) for r in R))
    print('max |p0-base| (reproducibility check)', max(abs(r['p0'] - r['base']) for r in R))
    for typ, ths in [('start', [0.8, 0.85, 0.9, 0.95]), ('stolen', [0.9, 0.95, 0.97, 0.99])]:
        for th in ths:
            for key in ['base', 'tta']:
                k = [r for r in R if r['typ'] == typ and r[key] >= th]
                c = Counter(r['lab'] for r in k)
                print('  %-6s th %.2f %-4s P %3d D %3d N %4d U %5d' % (typ, th, key, c['P'], c['D'], c['N'], c['U']))
