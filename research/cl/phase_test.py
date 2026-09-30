"""Does b1's node-level 'phase' head (trained on: GT node has two children) recognise mitosis?
(1) GT level: dividing GT nodes vs non-dividing GT nodes, per embryo, b1-seen vs b1-unseen movies.
(2) Candidate level: division-completion candidates (labels from cl2, B5 graphs): phase at p, q, p/q midpoint (t) and at a, b (t+1),
    combined with the fork prob; AUC for new divisions (P) vs non-divisions (N), D excluded.
usage: phase_test.py <gpu> <sets> -> /workspace/cl/phase/<set>__<movie>.json"""
import os, sys, json, glob
gpu = sys.argv[1]; os.environ['CUDA_VISIBLE_DEVICES'] = gpu
ART = '/workspace/art_b56/artifact_bundle'; sys.path.insert(0, ART); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import zarr
OUT = Path('/workspace/cl/phase'); OUT.mkdir(exist_ok=True)
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}


def main():
    from cell_event import SCALE, Movie, load_event_model
    import evalx
    model, _ = load_event_model(Path(ART) / 'b1_best.pt', 'cuda')
    rng = np.random.default_rng(0)

    def phase_of(mv, T, C):
        if not T: return np.zeros(0)
        X = torch.from_numpy(mv.patches(T, C)).cuda().float()
        with torch.inference_mode():
            E = torch.cat([model.encode(X[i:i + 512]).float() for i in range(0, len(X), 512)])
            return torch.sigmoid(model.phase(E).squeeze(-1)).cpu().numpy()
    for s in sys.argv[2].split(','):
        for f in sorted(glob.glob('/workspace/cl/cl2/%s/*.json' % s)):
            name = Path(f).stem; of = OUT / ('%s__%s.json' % (s, name))
            if of.exists(): continue
            mv = Movie(Path('/workspace/data/train') / (name + '.zarr'), context=5)
            # (1) GT nodes
            g = zarr.open_group('/workspace/data/train/%s.geff' % name, mode='r')
            ids = np.asarray(g['nodes/ids']).tolist(); T = np.asarray(g['nodes/props/t/values']).tolist()
            P = np.stack([np.asarray(g['nodes/props/%s/values' % k]) for k in 'zyx'], 1).tolist()
            E = np.asarray(g['edges/ids']).tolist(); nch = defaultdict(int)
            for a, b in E: nch[a] += 1
            div = [i for i in range(len(ids)) if nch[ids[i]] >= 2]
            non = [i for i in range(len(ids)) if nch[ids[i]] == 1]
            non = [non[j] for j in rng.choice(len(non), min(len(non), 60), replace=False)] if non else []
            sel = div + non
            ph = phase_of(mv, [T[i] for i in sel], [P[i] for i in sel])
            gt_rows = [dict(kind='gt', div=int(i in div), phase=float(p)) for i, p in zip(sel, ph)]
            # (2) candidates
            labs = [r for r in json.load(open(f)) if r['lab'] in ('P', 'N')]
            pos_ = [r for r in labs if r['lab'] == 'P']; neg = [r for r in labs if r['lab'] == 'N']
            keep = pos_ + ([neg[i] for i in rng.choice(len(neg), min(len(neg), 120), replace=False)] if neg else [])
            cand_rows = []
            if keep:
                from cell_event import chain, fork_geometry
                nodes, edges = evalx.load_graph_json(Path(B5[s]) / (name + '.json'))
                out_, prev_ = defaultdict(list), {}
                for e in edges:
                    a_, b_ = int(e['source_id']), int(e['target_id']); out_[a_].append(b_); prev_[b_] = a_
                pos3 = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
                raw = {n: [v['z'], v['y'], v['x']] for n, v in nodes.items()}
                TT, CC, GG = [], [], []
                for r in keep:
                    t = int(nodes[r['p']]['t']); q = r['q'] if r['q'] is not None else r['p']
                    vp = [(x + y) / 2 for x, y in zip(raw[r['p']], raw[q])]
                    TT += [t, t, t, t + 1, t + 1]; CC += [raw[r['p']], raw[q], vp, raw[r['a']], raw[r['b']]]
                    GG.append(fork_geometry(chain(r['p'], prev_, pos3), chain(r['a'], out_, pos3), chain(r['b'], out_, pos3)))
                X = torch.from_numpy(mv.patches(TT, CC)).cuda().float()
                with torch.inference_mode():
                    E = torch.cat([model.encode(X[i:i + 512]).float() for i in range(0, len(X), 512)]).view(len(keep), 5, -1)
                    ph = torch.sigmoid(model.phase(E).squeeze(-1)).cpu().numpy()
                    g = torch.from_numpy(np.stack(GG).astype(np.float32)).cuda()
                    fb = torch.sigmoid(model.fork_logits(E[:, 0], E[:, 3], E[:, 4], g)).cpu().numpy()
                    fv = torch.sigmoid(model.fork_logits(E[:, 2], E[:, 3], E[:, 4], g)).cpu().numpy()
                for r, p5, b0, bv in zip(keep, ph, fb, fv):
                    cand_rows.append(dict(kind='cand', typ=r['typ'], lab=r['lab'], base=float(b0), fork_vp=float(bv), ph_p=float(p5[0]), ph_q=float(p5[1]),
                                          ph_vp=float(p5[2]), ph_a=float(p5[3]), ph_b=float(p5[4]), n_neg_total=len(neg), n_neg_kept=len(keep) - len(pos_)))
            of.write_text(json.dumps(gt_rows + cand_rows))
    print('DONE', gpu, flush=True)


if __name__ == '__main__':
    main()
