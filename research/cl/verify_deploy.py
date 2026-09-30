"""check that the standalone deploy function removes exactly the nodes/edges of the evaluated variant {"border":1,"border_mode":"allyx"}"""
import sys, glob, os, json
from multiprocessing import Pool
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/code')


def one(f):
    import evalx
    from ideas import p19_edge as m, p19_edge_deploy as d
    name = os.path.basename(f)[:-5]
    nodes, edges = evalx.load_graph_json(f)
    shape = m._shape('/workspace/data/train/%s.zarr' % name)
    n1, e1, st = m.apply(nodes, edges, name=name, zarr='/workspace/data/train/%s.zarr' % name, border=1, border_mode='allyx')
    raw = json.load(open(f))  # string-keyed nodes, as in the P-stage JSON
    n2, e2, k = d.yx_border_stubs(raw['nodes'], raw['edges'], shape_yx=shape[-2:])
    ok = set(n1) == {int(x) for x in n2} and len(e1) == len(e2) and k == st['removed_nodes']
    return name, ok, k, shape


if __name__ == '__main__':
    fs = sorted(glob.glob('/workspace/cl/p16/ps_p17_*/graphs/*.json'))
    with Pool(4) as p: R = p.map(one, fs)
    print('movies', len(R), 'all identical', all(r[1] for r in R), 'removed', sum(r[2] for r in R),
          'shapes', sorted({tuple(r[3]) for r in R}), 'bad', [r[0] for r in R if not r[1]][:5])
