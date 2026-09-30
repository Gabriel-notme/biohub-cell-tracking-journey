"""dv_train_rule: LOEO models for the learned start-type division completion (daughter-birth) rule.
For each embryo E: LightGBM (b1 + structure, real start candidates + synthetic births from existing forks tp/fp) trained on E only;
thresholds chosen on E only: score at which E's greedy acceptance reaches r adds per movie (r = 0.05, 0.1, 0.2, 0.4).
Writes /workspace/cl/nm/dv_rule_<E>.txt (model) and dv_rule_thr.json; scores for every evaluable start candidate of the OTHER embryo
-> dv_rule_scores.json {movie: [[p, b, score], ...]}."""
import os, sys, json, glob
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, '/workspace/cl/nm')
from dv_probe import load, matrix, greedy, DROP
from dv_probe2 import fit


def loads(f):
    return [r for r in load(f) if r['typ'] == 0]


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'real+fork'
    files = sorted(glob.glob('/workspace/cl/nm/dv_cand/*.json'))
    with Pool(24) as p: rows = [r for rs in p.map(loads, files) for r in rs]
    ex = json.load(open('/workspace/cl/nm/dv_b1_feats.json'))
    for r in rows: r.update(ex['%s|%d|%d|%d' % (r['movie'], r['p'], r['a'], r['b'])])
    F = sorted({k for r in rows[:3000] for k in r} - DROP)
    F = [k for k in F if isinstance(next((r[k] for r in rows if r.get(k) is not None), 0), (int, float)) and not k.startswith('q')
         and k not in ('d_qp', 'd_qb', 'Iq', 'Cq', 'typ') and k[:1] not in 'ISMCr']
    print('features', F)
    E = {e: [r for r in rows if r['emb'] == e] for e in ['44b6', '6bba']}
    nmov = {e: len({r['movie'] for r in rows if r['emb'] == e}) for e in E}
    thr, scores = {}, {}
    for e, o in [('44b6', '6bba'), ('6bba', '44b6')]:
        m, npos = fit(E[e], F, mode)
        m.save_model('/workspace/cl/nm/dv_rule_%s.txt' % e)
        own = [r for r in E[e] if r['src'] == 'cand']; so = m.predict(matrix(own, F))
        _, hist = greedy(own, so, ks=(1, 400))
        o_hist = sorted(so, reverse=True)
        thr[e] = {}
        for rr in [0.05, 0.1, 0.2, 0.4]:
            k = max(1, int(round(rr * nmov[e])))
            thr[e][str(rr)] = float(hist[min(k, len(hist)) - 1][2]) if hist else 1.0
        # in-sample precision at those thresholds on the training embryo (optimistic)
        for rr, t in thr[e].items():
            acc = [h for h in hist if h[2] >= t]
            print('train %s r %s thr %.4f in-sample greedy TP/FP %s' % (e, rr, t, acc[-1][:2] if acc else (0, 0)))
        te = [r for r in E[o] if r['src'] == 'cand']; st = m.predict(matrix(te, F))
        for r, s in zip(te, st): scores.setdefault(r['movie'], []).append([r['p'], r['b'], float(s), r['lab'], r['set']])
        for rr, t in thr[e].items():
            sel = [(r, s) for r, s in zip(te, st) if s >= t]
            print('  -> apply to %s (thr from %s, r %s): candidates above thr %d (P %d); clean40 above %d (P %d)' % (
                o, e, rr, len(sel), sum(r['lab'] == 'P' for r, s in sel), sum(r['set'] in ('hold36', 'prev4') for r, s in sel),
                sum(r['lab'] == 'P' and r['set'] in ('hold36', 'prev4') for r, s in sel)))
    json.dump(thr, open('/workspace/cl/nm/dv_rule_thr.json', 'w'))
    json.dump(scores, open('/workspace/cl/nm/dv_rule_scores.json', 'w'))
