"""Sanity check for evt_tta on one movie: identity view == original embeddings; per-view edge logits agree better when the
geometry is transformed with the view than when it is not (checks the sign/swap convention); timing of the embedding step.
usage: tta_check.py <set> <movie> [mode]"""
import os, sys, json, time
import numpy as np
os.environ.setdefault('BIOHUB_ART', '/workspace/art_b56/artifact_bundle')
ART = '/workspace/art_b56/artifact_bundle'; sys.path.insert(0, ART); sys.path.insert(0, '/workspace/cl')
RUNW = {'hold36': '/workspace/runs/b5f_hold36/working', 'prev4': '/workspace/runs/b5f_prev4/working', 'audit32': '/workspace/sync3/runs/b5f_audit32/working',
        't127a': '/workspace/sync4/runs/b5f_t127a/working', 't127b': '/workspace/sync3/runs/b5f_t127b/working'}
s, name = sys.argv[1], sys.argv[2]; mode = sys.argv[3] if len(sys.argv) > 3 else 'd4'
os.environ['BIOHUB_BASE_REPO'] = RUNW[s] + '/tracking_repo'
import torch
from refine_events import EventRefiner, structure, batch_edge_geometry
import evt_tta
sel = json.load(open(ART + '/final_selection.json'))['B5']['base']
d = json.load(open(RUNW[s] + '/reference_graphs/%s.json' % name)); nodes = {int(k): v for k, v in d['nodes'].items()}; edges = d['edges']
ref = EventRefiner([ART + '/' + n for n in sel['model_files']], '/workspace/data/train', sel['config'], '/workspace/cache/tta_check_orig')
t0 = time.time(); ids0, emb0 = ref.embeddings(name, nodes); t_orig = time.time() - t0
evt_tta.install(); os.environ['EVT_TTA'] = mode
ref2 = EventRefiner([ART + '/' + n for n in sel['model_files']], '/workspace/data/train', sel['config'], '/workspace/cache/tta_check_' + mode)
t0 = time.time(); ids1, emb1 = ref2.embeddings(name, nodes); t_tta = time.time() - t0
print('nodes', len(ids0), 'emb sec orig %.1f tta %.1f' % (t_orig, t_tta))
for j in range(len(emb0)): print('model', j, 'identity max abs diff', float(np.abs(emb0[j] - emb1[j]).max()), 'scale', float(np.abs(emb0[j]).mean()))
out, prev, frames, pos = structure(nodes, edges)
rows = [(int(e['source_id']), int(e['target_id'])) for e in edges][:20000]
g = batch_edge_geometry(rows, prev, out, pos); lookup = {n: i for i, n in enumerate(ids1)}
allv = [evt_tta._VIEWS[id(b)][1] for b in emb1]; V = evt_tta.views(mode)
ix = np.asarray([[lookup[a], lookup[b]] for a, b in rows])
for j, model in enumerate(ref2.models):
    def lg(vi, geo):
        with torch.inference_mode():
            z = [torch.from_numpy(allv[j][vi][ix[:, i]]).cuda() for i in range(2)]
            return model.edge_logits(*z, torch.from_numpy(np.ascontiguousarray(geo)).cuda()).float().cpu().numpy()
    base = lg(0, g)
    for vi, v in enumerate(V[1:], 1):
        a = lg(vi, evt_tta.tg(g, v)); b = lg(vi, g)
        print('model', j, 'view', v, 'mean|dlogit| transformed-geo %.3f  untransformed-geo %.3f  corr %.3f' % (np.abs(a - base).mean(), np.abs(b - base).mean(), np.corrcoef(a, base)[0, 1]))
p0 = ref.score('edge', rows, g, emb0, {n: i for i, n in enumerate(ids0)}); p1 = ref2.score('edge', rows, g, emb1, lookup)
print('edge prob orig vs tta: corr %.4f, frac |dp|>0.1 %.4f, n<0.7 orig %d tta %d' % (np.corrcoef(p0, p1)[0, 1], np.mean(np.abs(p0 - p1) > .1), (p0 < .7).sum(), (p1 < .7).sum()))
