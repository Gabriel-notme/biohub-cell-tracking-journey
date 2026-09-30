"""Division-completion forks in P13 outputs split by type: 'start' (second daughter had no parent in the B5 graph) vs 'stolen'
(second daughter was taken from another parent). TP/FP/U per embryo and clean/in-sample split (TTA probe rows carry labels)."""
import json, glob
from collections import Counter, defaultdict
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}
rows = []
for f in glob.glob('/workspace/cl/ttaf/*.json'):
    s, m = f.split('/')[-1][:-5].split('__')
    R = [r for r in json.load(open(f)) if r['dc']]
    if not R: continue
    g = json.load(open('/workspace/cl/ps_p13_%s/graphs/%s.json' % (s, m)))
    b = json.load(open('%s/%s.json' % (B5[s], m)))
    par_b5 = {int(e['target_id']): int(e['source_id']) for e in b['edges']}
    ch = defaultdict(list); eat = {}
    for e in g['edges']:
        a_, b_ = int(e['source_id']), int(e['target_id']); ch[a_].append(b_); eat[(a_, b_)] = e
    for r in R:
        p = r['p']; kids = ch[p]
        dcs = [k for k in kids if eat[(p, k)].get('div_complete')]
        if not dcs: continue
        k = dcs[0]
        typ = 'start' if k not in par_b5 else ('stolen' if par_b5[k] != p else 'other')
        rows.append((s, m[:4], typ, r['lab']))
for grp, sets in [('clean', ('hold36', 'prev4')), ('train', ('t127a', 't127b', 'audit32'))]:
    for emb in ['44b6', '6bba']:
        for typ in ['start', 'stolen']:
            c = Counter(l for s, e, t, l in rows if s in sets and e == emb and t == typ)
            print('%-5s %s %-6s TP %2d FP %2d U %4d' % (grp, emb, typ, c['TP'], c['FP'], c['U']))
print('other types', Counter(t for s, e, t, l in rows))
