import zarr, json, numpy as np
g = zarr.open_group('/workspace/runs/fullgraph_hold36/44b6_c8e2a523.geff', mode='r')
def walk(grp, pre=''):
    for k in grp.keys():
        x = grp[k]
        if hasattr(x, 'shape'): print(pre + k, x.shape, x.dtype)
        else: walk(x, pre + k + '/')
walk(g)
ids = np.asarray(g['nodes/ids'][:])
d = json.load(open('/workspace/runs/p3_hold36/graphs/44b6_c8e2a523.json'))
fin = set(int(k) for k in d['nodes'])
print('full nodes', len(ids), 'final nodes', len(fin), 'final in full', len(fin & set(ids.tolist())))
e = np.asarray(g['edges/ids'][:]); print('full edges', e.shape)
fe = set((int(x['source_id']), int(x['target_id'])) for x in d['edges'])
print('final edges', len(fe), 'in full', len(fe & set(map(tuple, e.tolist()))))
print(list(d['nodes'].values())[0]); print(d['edges'][0])
