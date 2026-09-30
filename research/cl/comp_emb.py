"""Per-embryo, per-set decomposition of P13 (== P11 on labelled data): B5 -> [division completion + dfork K=35/100 + isolated-node prune]
-> [+ learned relink + free-end linking]. 'clean' = hold36+prev4 (never used to train the link models); the other sets are in-sample for them."""
import json, sys
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
b5 = {r['movie']: r for r in json.load(open('/workspace/cl/rev/b5all.json'))}
lo = json.load(open('/workspace/cl/lo/eval_base_none-dep-cross-in_0.65_0.4.json'))
none = {r['movie']: r for r in lo if r['arm'] == 'none'}; dep = {r['movie']: r for r in lo if r['arm'] == 'dep'}; cross = {r['movie']: r for r in lo if r['arm'] == 'cross'}
sets = {r['movie']: r['set'] for r in lo}
S = lambda d, ms: summarise([d[m] for m in ms])
print('%-22s %8s %8s %8s %8s | %s' % ('group', 'B5', '+div', '+links', '+linksX', 'div TP/FP/FN B5 -> +div'))
for nm, f in [('44b6 clean', lambda m: m.startswith('44b6') and sets[m] in ('hold36', 'prev4')), ('6bba clean', lambda m: m.startswith('6bba') and sets[m] in ('hold36', 'prev4')),
              ('44b6 all71', lambda m: m.startswith('44b6')), ('6bba all128', lambda m: m.startswith('6bba')), ('all199', lambda m: True)]:
    ms = [m for m in none if f(m)]
    a, b, c, d = S(b5, ms), S(none, ms), S(dep, ms), S(cross, ms)
    print('%-22s %.5f %.5f %.5f %.5f | %d/%d/%d -> %d/%d/%d   gains: div %+.4f links(dep) %+.4f links(cross) %+.4f' % (
        nm, a['score'], b['score'], c['score'], d['score'], a['division_tp'], a['division_fp'], a['division_fn'], b['division_tp'], b['division_fp'], b['division_fn'],
        b['score'] - a['score'], c['score'] - b['score'], d['score'] - b['score']))
