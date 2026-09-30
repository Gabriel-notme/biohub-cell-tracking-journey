"""Deltas of each variant vs variant 1 (the P14 reference) from rule_eval_last_<mod>.json, per embryo + bootstrap + clean40."""
import sys, json
sys.path.insert(0, '/workspace/official/src')
import numpy as np
from tracking_cellmot.metrics import summarise
R = json.load(open(sys.argv[1])); ref = int(sys.argv[2]) if len(sys.argv) > 2 else 1
nv = max(r['vi'] for r in R) + 1
base = {r['movie']: r for r in R if r['vi'] == ref}; ms = sorted(base)
rng = np.random.default_rng(0); K = [rng.integers(0, len(ms), len(ms)) for _ in range(300)]
for i in range(nv):
    if i == ref: continue
    cur = {r['movie']: r for r in R if r['vi'] == i}
    line = 'v%d' % i
    for emb in ['44b6', '6bba', '']:
        mm = [m for m in ms if m.startswith(emb)]
        a = summarise([base[m] for m in mm]); b = summarise([cur[m] for m in mm])
        line += ' | %s %+.5f (e %+.5f, div %+d/%+d)' % (emb or 'all', b['score'] - a['score'], b['adj_edge_jaccard'] - a['adj_edge_jaccard'], b['division_tp'] - a['division_tp'], b['division_fp'] - a['division_fp'])
    bs = [summarise([cur[ms[j]] for j in k])['score'] - summarise([base[ms[j]] for j in k])['score'] for k in K]
    line += ' | CI [%+.5f, %+.5f]' % (np.quantile(bs, .025), np.quantile(bs, .975))
    cm = [m for m in ms if base[m]['set'] in ('hold36', 'prev4')]
    line += ' | clean40 %+.5f' % (summarise([cur[m] for m in cm])['score'] - summarise([base[m] for m in cm])['score'])
    line += ' | nodes %+d' % (sum(cur[m]['num_pred_nodes'] for m in ms) - sum(base[m]['num_pred_nodes'] for m in ms))
    st = {}
    for m in ms:
        for k2, v in cur[m].items():
            if k2.startswith('cut') or k2 == 'dropped': st[k2] = st.get(k2, 0) + v
    print(line, st, flush=True)
