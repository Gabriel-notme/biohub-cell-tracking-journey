"""P17: deletion-type structure clean-up on the final P15 graph (GT-free; only removes implausible structure):
  cutdup     : remove surviving B5 gap bridges (gap_closed / gap2) whose synthetic node lies within 3.2 um of another node of its frame
               (r4_gapre mode 'cutdup').
  forkfrag   : remove weakly connected components with < 6 nodes and no fork that contain a former B5 fork daughter which now has
               no parent (fragments B5's own short-track filter never re-checked; r4_forkfrag src='ref').
  shortbranch: a fork whose daughter branch ends within 2 frames (<= 3 nodes, no further fork) is not a division: delete that branch.
Each step is kept only if the graph stays valid."""
import sys
from collections import defaultdict
sys.path.insert(0, '/workspace/cl/ideas')
WANTS_META = True


def short_branch(nodes, edges, maxk=2):
    out = defaultdict(list)
    for e in edges: out[int(e['source_id'])].append(int(e['target_id']))
    drop = set()
    for p, ch in list(out.items()):
        if len(ch) != 2: continue
        for c in ch:
            br = [c]; n = c; ok = True
            while True:
                nx = out.get(n, [])
                if len(nx) == 0: break
                if len(nx) == 2 or len(br) > maxk: ok = False; break
                n = nx[0]; br.append(n)
            if ok and len(br) <= maxk + 1:
                drop.update(br); break  # one branch per fork
    if not drop: return nodes, edges, {'sb_forks': 0, 'sb_nodes': 0}
    nn = {k: v for k, v in nodes.items() if k not in drop}
    ne = [e for e in edges if int(e['source_id']) not in drop and int(e['target_id']) not in drop]
    return nn, ne, {'sb_nodes': len(drop)}


def apply(nodes, edges, cutdup=1, forkfrag=1, shortbranch=1, **meta):
    st = {}
    if cutdup:
        import r4_gapre
        nodes, edges, s = r4_gapre.apply(nodes, edges, mode='cutdup', **meta); st.update({'cd_' + k: v for k, v in s.items()})
    if forkfrag:
        import r4_forkfrag
        nodes, edges, s = r4_forkfrag.apply(nodes, edges, src='ref', rescue=0, minlen=6, **meta); st.update({'ff_' + k: v for k, v in s.items()})
    if shortbranch:
        nodes, edges, s = short_branch(nodes, edges); st.update(s)
    return nodes, edges, st
