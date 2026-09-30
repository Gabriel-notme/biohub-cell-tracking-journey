"""Do dense movies prefer different linking thresholds (cross-embryo arm)? Per embryo x density bin, score for each threshold pair."""
import json, glob, sys
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
runs = {}
for f in glob.glob('/workspace/cl/lo/eval_base_cross_*.json') + ['/workspace/cl/lo/eval_base_none-dep-cross-in_0.65_0.4.json']:
    R = json.load(open(f))
    for r in R:
        if r['arm'] != 'cross': continue
        runs.setdefault(tuple(r['th']), {})[r['movie']] = r
none = {r['movie']: r for r in json.load(open('/workspace/cl/lo/eval_base_none-dep-cross-in_0.65_0.4.json')) if r['arm'] == 'none'}
ths = sorted(runs)
print('thresholds', ths)
for emb in ['44b6', '6bba']:
    for lo, hi in [(0, 10000), (10000, 30000), (30000, 1e9)]:
        ms = [m for m in none if m.startswith(emb) and lo <= none[m]['num_pred_nodes'] < hi]
        if not ms: continue
        base = summarise([none[m] for m in ms])['score']
        print('%s npred %6d-%6d n=%3d: ' % (emb, lo, min(hi, 999999), len(ms)) + ' '.join('%s %+.4f' % ('/'.join('%g' % x for x in t), summarise([runs[t][m] for m in ms])['score'] - base) for t in ths))
