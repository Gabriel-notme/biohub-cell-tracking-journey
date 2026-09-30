import os, sys, json
sys.path.insert(0, '/workspace/p17ds'); import p17_post
import pandas as pd
df15 = pd.read_csv('/workspace/kout/p15/submission.csv'); df17 = pd.read_csv('/workspace/kout/p17/submission.csv')
for m in sorted(df17.dataset.unique()):
    d = json.load(open(f'/workspace/kout/p15/pstage_graphs/{m}.json')); nodes = {int(k): v for k, v in d['nodes'].items()}
    n2, e2, st = p17_post.apply(nodes, d['edges'], f'/workspace/kout/p15/reference_graphs/{m}.json')
    a = df15[(df15.dataset == m) & (df15.row_type == 'node')]; b = df17[(df17.dataset == m) & (df17.row_type == 'node')]
    ida = set(a.node_id.astype(int)); idb = set(b.node_id.astype(int))
    gi = set(nodes); gi2 = set(n2)
    print(m, 'csv15 ids==graph15 ids', ida == gi, 'csv17 ids==replay ids', idb == gi2, 'removed ids equal', (ida - idb) == (gi - gi2), len(ida - idb))
    # edges by id
    ea = set(zip(df17[(df17.dataset == m) & (df17.row_type == 'edge')].source_id.astype(int), df17[(df17.dataset == m) & (df17.row_type == 'edge')].target_id.astype(int)))
    er = {(int(e['source_id']), int(e['target_id'])) for e in e2}
    print('   csv17 edges == replay edges', ea == er, len(ea ^ er))
    # coordinate mismatch sample
    mm = 0; ex = []
    for r in a.itertuples():
        v = nodes.get(int(r.node_id))
        if v is None: continue
        if (int(r.z), int(r.y), int(r.x)) != (round(v['z']), round(v['y']), round(v['x'])):
            mm += 1
            if len(ex) < 2: ex.append(((r.z, r.y, r.x), (v['z'], v['y'], v['x'])))
    print('   coord mismatches vs round():', mm, ex)
