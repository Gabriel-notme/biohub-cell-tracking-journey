"""Fault injection: execute the EXACT p17 / p19small blocks of /workspace/p19ds/p_stage12.py (text-extracted, dedented) on a
P15 graph under several failure scenarios and check the fallback graph + validity."""
import sys, shutil, textwrap, types, importlib
sys.path.insert(0, '/workspace/cl/p16/redteam_p20')
from rt_common import *
src = Path('/workspace/p19ds/p_stage12.py').read_text().split('\n')
i0 = next(i for i, l in enumerate(src) if "if cfg.get('p17') is not None:" in l)
i1 = next(i for i, l in enumerate(src) if i > i0 and l.strip().startswith("pairs = [(int(e['source_id'])"))
BLOCK = textwrap.dedent('\n'.join(src[i0:i1]))
ns_stage = {}
exec(compile(textwrap.dedent('\n'.join(src[:next(i for i, l in enumerate(src) if l.startswith('def worker'))])), 'p_stage12_head', 'exec'), ns_stage)
check_graph = ns_stage['check_graph']
cfg = json.loads(Path('/workspace/p19ds/p20_config.json').read_text())
FI = Path('/workspace/cl/p16/redteam_p20/fi')


def run_block(nodes_in, edges_in, graphs_dir, data_dir, name):
    a = types.SimpleNamespace(graphs=str(graphs_dir), data=str(data_dir))
    g = dict(cfg=cfg, Path=Path, a=a, name=name, nodes2=nodes_in, ne=edges_in, nodes=nodes_in, st={}, check_graph=check_graph)
    t0 = time.perf_counter(); exec(BLOCK, g); dt = time.perf_counter() - t0
    return g['nodes2'], g['ne'], g['st'], dt


def mkcase(case, name, ref_src=None, ref_text=None):
    d = FI / case; shutil.rmtree(d, ignore_errors=True); (d / 'graphs').mkdir(parents=True); (d / 'reference_graphs').mkdir()
    if ref_src is not None: (d / 'reference_graphs' / (name + '.json')).symlink_to(ref_src)
    if ref_text is not None: (d / 'reference_graphs' / (name + '.json')).write_text(ref_text)
    return d / 'graphs'


def mkzarr(case, name, shape):
    import zarr
    d = FI / ('data_' + case); shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    g = zarr.open_group(str(d / (name + '.zarr')), mode='w'); g.create_array('0', shape=shape, chunks=(1,) + tuple(shape[1:]), dtype='uint16')
    return d


def summary(tag, n0, e0, n, e, st, dt, T, ref=None):
    F0 = forks(e0); F = forks(e)
    lost = sum(1 for p, c in F0.items() if F.get(p) != c)
    same_as_input = set(n) == set(n0) and set(pairs(e)) == set(pairs(e0))
    extra = ''
    if ref is not None: extra = ' equal_to_normal=%s' % (set(n) == set(ref[0]) and set(pairs(e)) == set(pairs(ref[1])))
    print('%-22s del=%d fork_lost=%d unchanged=%s valid=%s t=%.2fs st=%s%s' % (tag, len(n0) - len(n), lost, same_as_input, validity(n, e, T) or 'OK', dt,
          {k: (v[:90] if isinstance(v, str) else v) for k, v in st.items()}, extra))


for s, m in [('hold36', '44b6_7a302da0'), ('t127a', '6bba_57b7cc1e') if (Path('/workspace/cl/ps_p15_t127a/graphs/6bba_57b7cc1e.json')).exists() else ('t127b', '6bba_57b7cc1e')]:
    if not p15p(s, m).exists():
        s = next(ss for ss in SETS if p15p(ss, m).exists())
    print('==', s, m)
    n15, e15, _ = load(p15p(s, m)); shp = zshape(s, m); T = shp[0]; rp = refp(s, m); data = Path('/workspace/cl/data_' + s)
    n19, e19, _ = load(p19p(s, m))
    # A normal
    nA, eA, stA, dt = run_block(n15, e15, mkcase('A', m, ref_src=rp), data, m); summary('A normal', n15, e15, nA, eA, stA, dt, T, ref=(n19, e19))
    # B reference graph missing
    n, e, st, dt = run_block(n15, e15, mkcase('B', m), data, m); summary('B ref missing', n15, e15, n, e, st, dt, T)
    # C reference graph corrupted
    n, e, st, dt = run_block(n15, e15, mkcase('C', m, ref_text='{"nodes": {}'), data, m); summary('C ref corrupt', n15, e15, n, e, st, dt, T)
    # C2 reference graph with no edges key
    n, e, st, dt = run_block(n15, e15, mkcase('C2', m, ref_text='{"nodes": {}}'), data, m); summary('C2 ref no edges', n15, e15, n, e, st, dt, T)
    # D reference graph of another movie (id space mismatch)
    other = next(p for p in sorted(rp.parent.glob('*.json')) if p.stem != m)
    n, e, st, dt = run_block(n15, e15, mkcase('D', m, ref_src=other), data, m); summary('D ref other movie', n15, e15, n, e, st, dt, T, ref=(n19, e19))
    # E zarr missing
    n, e, st, dt = run_block(n15, e15, mkcase('E', m, ref_src=rp), FI / 'nonexistent_data', m); summary('E zarr missing', n15, e15, n, e, st, dt, T)
    # F zarr shape 512x512 / 64x64 / 5-D
    for tag, shape in [('F big 512', (T, 64, 512, 512)), ('F small 64', (T, 64, 64, 64)), ('F 5D', (T, 1, 64, 256, 256))]:
        n, e, st, dt = run_block(n15, e15, mkcase('F', m, ref_src=rp), mkzarr('F', m, shape), m); summary(tag, n15, e15, n, e, st, dt, T, ref=(n19, e19))
    # G rule raising
    import p17_post, p19_dup, p19_edge_deploy
    for modn, fn in [('p17_post', 'forkfrag'), ('p19_dup', 'par_dup'), ('p19_edge_deploy', 'yx_border_stubs')]:
        mod = sys.modules[modn]; orig = getattr(mod, fn)
        def boom(*x, **k): raise RuntimeError('injected ' + fn)
        setattr(mod, fn, boom)
        try:
            n, e, st, dt = run_block(n15, e15, mkcase('A', m, ref_src=rp), data, m); summary('G raise ' + fn, n15, e15, n, e, st, dt, T)
        finally:
            setattr(mod, fn, orig)
    # H rule returns a graph that empties a frame -> check_graph must reject
    orig = p19_edge_deploy.yx_border_stubs
    def empty_frame(nodes, edges, shape_yx=None):
        drop = {k for k, v in nodes.items() if int(v['t']) == 50}
        return {k: v for k, v in nodes.items() if k not in drop}, [x for x in edges if int(x['source_id']) not in drop and int(x['target_id']) not in drop], len(drop)
    p19_edge_deploy.yx_border_stubs = empty_frame
    try:
        n, e, st, dt = run_block(n15, e15, mkcase('A', m, ref_src=rp), data, m); summary('H frame emptied', n15, e15, n, e, st, dt, T)
    finally:
        p19_edge_deploy.yx_border_stubs = orig
    # I dangling edge produced by a rule -> check_graph must reject
    orig = p19_dup.par_dup
    def dangling(nodes, edges, **k):
        victim = next(iter(nodes)); return {kk: v for kk, v in nodes.items() if kk != victim}, edges, 1, 0
    p19_dup.par_dup = dangling
    try:
        n, e, st, dt = run_block(n15, e15, mkcase('A', m, ref_src=rp), data, m); summary('I dangling', n15, e15, n, e, st, dt, T)
    finally:
        p19_dup.par_dup = orig
    # J input graph not mutated by the block
    n15b, e15b, _ = load(p15p(s, m))
    print('J input untouched:', json.dumps({str(k): v for k, v in n15.items()}, sort_keys=True) == json.dumps({str(k): v for k, v in n15b.items()}, sort_keys=True) and json.dumps(e15, sort_keys=True) == json.dumps(e15b, sort_keys=True))
    # K timing incl. check_graph: normal block 3x
    ts = [run_block(n15, e15, mkcase('A', m, ref_src=rp), data, m)[3] for _ in range(3)]
    print('K block time (exact p_stage code incl. imports, ref loads, zarr read, check_graph):', [round(x, 2) for x in ts])
shutil.rmtree(FI, ignore_errors=True)
