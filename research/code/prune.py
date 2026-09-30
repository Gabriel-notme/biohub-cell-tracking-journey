from collections import defaultdict

def prune_fragments(nodes, edges, min_len=2):
    out = defaultdict(list); par = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); out[s].append(d); par[d] = s
    # weakly connected components
    adj = defaultdict(list)
    for s, ds in out.items():
        for d in ds: adj[s].append(d); adj[d].append(s)
    seen = set(); drop = set()
    for n in nodes:
        if n in seen: continue
        comp = []; st = [n]; seen.add(n)
        while st:
            x = st.pop(); comp.append(x)
            for y in adj.get(x, []):
                if y not in seen: seen.add(y); st.append(y)
        if len(comp) < min_len and not any(len(out.get(x, [])) > 1 for x in comp):
            drop.update(comp)
    nn = {k: v for k, v in nodes.items() if k not in drop}
    ne = [e for e in edges if int(e['source_id']) not in drop and int(e['target_id']) not in drop]
    # keep every frame represented (export requires all frames)
    ts = {int(v['t']) for v in nodes.values()}; ts2 = {int(v['t']) for v in nn.values()}
    if ts != ts2:
        for k, v in nodes.items():
            if int(v['t']) in ts - ts2: nn[k] = v; ts2.add(int(v['t']))
    return nn, ne, {'pruned_nodes': len(nodes) - len(nn)}
