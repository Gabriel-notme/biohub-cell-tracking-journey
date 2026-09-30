"""P19 'edge' post-step (GT-free, deletion only; runs after P17's cutdup/forkfrag/shortbranch on the final graph).
yxborder: remove every weakly connected component with < 6 nodes (B5 OUTPUT_MIN_TRACK_LEN) and no fork whose EVERY node lies within
2 voxels of the y or x edge of the field of view (y <= 2 or y >= Y-3 or x <= 2 or x >= X-3, voxel coordinates). These are partial
nuclei cut by the lateral FOV border that B5's lineage stage split off into fragments; they are never part of a division.
Strict-validated on P17 (199 movies): +0.00006, CI [+0.00004, +0.00010], 1489 nodes removed, division TP/FP 0/0.
Pure python; nodes = {id: {'t','z','y','x',...}}, edges = [{'source_id','target_id',...}]; shape_yx = (Y, X) of the image
(image.shape[-2:], 256 x 256 for this competition)."""
from collections import defaultdict

MINLEN = 6       # B5 OUTPUT_MIN_TRACK_LEN
MARGIN = 2.0     # voxels from the y/x image edge


def yx_border_stubs(nodes, edges, shape_yx=(256, 256), minlen=MINLEN, margin=MARGIN):
    Y, X = int(shape_yx[0]), int(shape_yx[1])
    key = {int(k): k for k in nodes}
    adj = defaultdict(list); out = defaultdict(int)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); adj[a].append(b); adj[b].append(a); out[a] += 1

    def isb(n):
        v = nodes[key[n]]; y, x = float(v['y']), float(v['x'])
        return y <= margin or y >= Y - 1 - margin or x <= margin or x >= X - 1 - margin

    seen = set(); drop = set()
    for n in key:
        if n in seen: continue
        comp = [n]; st = [n]; seen.add(n)
        while st:
            u = st.pop()
            for w in adj.get(u, []):
                if w not in seen: seen.add(w); st.append(w); comp.append(w)
        if len(comp) >= minlen or any(out.get(u, 0) >= 2 for u in comp): continue
        if all(isb(u) for u in comp): drop.update(comp)
    if not drop: return nodes, edges, 0
    return ({k: v for k, v in nodes.items() if int(k) not in drop},
            [e for e in edges if int(e['source_id']) not in drop and int(e['target_id']) not in drop], len(drop))
