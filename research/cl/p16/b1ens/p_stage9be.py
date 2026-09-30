"""P-series post-lineage stage (v3 + P14 post-steps term_trim / long_link, see p14_post.py, + P15 tbext + relinefit, see p15_post.py): division completion on the final lineage graph, then learned relinking and
learned free-end linking driven by the saved pre-ILP candidate graph.

Adds forks where the frozen image-conditioned event model (b1 fork head) gives a high
division probability to a parent whose second daughter currently starts a new track
(and, optionally, where the second daughter is held by a weakly supported parent).
Every movie falls back to its unmodified graph on any error.
"""
from pathlib import Path
import os, sys, json, time, argparse, hashlib, subprocess, csv, traceback
from collections import Counter, defaultdict

def load_cfg(a):
    return json.loads(Path(a.config).read_text())

def check_graph(nodes, edges, orig_nodes):
    pairs = [(int(e['source_id']), int(e['target_id'])) for e in edges]
    assert len(pairs) == len(set(pairs)), 'dup edges'
    assert max(Counter(t for s, t in pairs).values(), default=0) <= 1, 'merge'
    assert max(Counter(s for s, t in pairs).values(), default=0) <= 2, 'outdeg'
    assert all(s in nodes and t in nodes and int(nodes[t]['t']) == int(nodes[s]['t']) + 1 for s, t in pairs), 'dangling/nonconsecutive'
    assert {int(v['t']) for v in nodes.values()} == {int(v['t']) for v in orig_nodes.values()}, 'frames'

def worker(a):
    os.environ['BIOHUB_ART'] = a.artifact
    import numpy as np, torch
    sys.path.insert(0, a.artifact); sys.path.insert(0, '/workspace/p56stage')
    from refine_events import EventRefiner
    import div_complete as dc
    sys.path.insert(0, '/workspace/cl/p16/b1ens'); import b1e_patch; b1e_patch.install(dc); print('DN_INSTALLED', os.environ.get('DN_MODE'), os.environ.get('DN_MAP'), os.environ.get('DN_B1LOEO'), flush=True)
    cfg = load_cfg(a)
    names = json.loads(Path(a.work, 'pstage_shard_%d.json' % a.worker).read_text())
    er = EventRefiner([Path(a.artifact) / n for n in cfg['models']], a.data, cfg.get('event_config', {'fork_ensemble': 'first', 'edge_ensemble': 'last'}), Path(a.work) / ('pstage_cache_%d' % a.worker))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = Path(a.graphs) / (name + '.json'); dst = out / (name + '.json')
        d = json.loads(src.read_text())
        if float(a.deadline) > 0 and time.time() > float(a.deadline):
            tmp = dst.with_name(name + '.%d.tmp' % os.getpid())
            tmp.write_text(json.dumps({'nodes': d['nodes'], 'edges': d['edges'], 'pstage': {'skipped': 'deadline'}})); os.replace(tmp, dst)
            print('PSTAGE_SKIP', name, flush=True); continue
        nodes = {int(k): v for k, v in d['nodes'].items()}; edges = d['edges']
        t0 = time.time(); stats = {}
        try:
            fullp0 = Path(a.full) / (name + '.geff') if a.full else None
            if (cfg.get('pre_relink') is not None or cfg.get('pre_edge_link') is not None) and fullp0 is not None and fullp0.exists():
                cdir0 = Path(a.config).parent
                try:
                    import relink, edge_link
                    n0, e0 = nodes, edges
                    if cfg.get('pre_relink') is not None:
                        n0, e0, s0 = relink.apply(n0, e0, edge_link.load_full(fullp0), cdir0 / cfg['pre_relink'].get('model', 'relink_lgb.json'), th=float(cfg['pre_relink']['th']))
                        check_graph(n0, e0, nodes); stats['pre_rl'] = s0.get('rl_applied', 0)
                    if cfg.get('pre_edge_link') is not None:
                        el0 = cfg['pre_edge_link']
                        n0, e0, s0 = edge_link.apply(n0, e0, fullp0, cdir0 / el0.get('model', 'edge_lgb.json'), th=float(el0['th']), allow_gap2=bool(el0.get('gap2', True)))
                        check_graph(n0, e0, nodes); stats['pre_el'] = s0.get('el_gap1', 0) + s0.get('el_gap2', 0)
                    nodes, edges = n0, e0
                except Exception as e:
                    stats['pre_link_error'] = repr(e)[:300]
            cands = dc.score_candidates(er, name, nodes, edges)
            ne, st = dc.apply(nodes, edges, cands, **cfg['apply'])
            nodes2 = nodes
            fullp = Path(a.full) / (name + '.geff') if a.full else None
            if cfg.get('dsr') is not None and fullp is not None and fullp.exists():
                try:
                    import dsr
                    n3, e3, st3 = dsr.recover(er, name, nodes2, ne, fullp, **cfg['dsr'])
                    check_graph(n3, e3, nodes)
                    nodes2, ne = n3, e3; st = dict(st, **st3)
                except Exception as e:
                    st = dict(st, dsr_error=repr(e)[:300])
            elif cfg.get('dsr') is not None:
                st = dict(st, dsr_skipped='no_full_graph')
            if cfg.get('dfork') is not None:
                try:
                    import dfork
                    e4, st4 = dfork.resolve(nodes2, ne, **cfg['dfork'])
                    check_graph(nodes2, e4, nodes)
                    ne = e4; st = dict(st, **st4)
                except Exception as e:
                    st = dict(st, dfork_error=repr(e)[:300])
            if cfg.get('prune_min_len'):
                import prune
                nodes2, ne, pst = prune.prune_fragments(nodes2, ne, int(cfg['prune_min_len'])); st = dict(st, **pst)
            ri_new = None; ri_backup = (nodes2, ne)
            if cfg.get('frag_reinsert') is not None and fullp is not None and fullp.exists():
                try:
                    import frag_reinsert, edge_link
                    fr = cfg['frag_reinsert']
                    n9, e9, ri_new, st9 = frag_reinsert.add(nodes2, ne, edge_link.load_full(fullp), max_len=int(fr.get('max_len', 5)), clear_um=float(fr.get('clear_um', 4.0)))
                    nodes2, ne = n9, e9; st = dict(st, **st9)
                except Exception as e:
                    ri_new = None; nodes2, ne = ri_backup; st = dict(st, ri_error=repr(e)[:300])
            cdir = Path(a.config).parent
            if cfg.get('relink') is not None and fullp is not None and fullp.exists():
                try:
                    import relink, edge_link
                    n5, e5, st5 = relink.apply(nodes2, ne, edge_link.load_full(fullp), cdir / cfg['relink'].get('model', 'relink_lgb.json'), th=float(cfg['relink']['th']))
                    check_graph(n5, e5, nodes)
                    nodes2, ne = n5, e5; st = dict(st, **st5)
                except Exception as e:
                    st = dict(st, relink_error=repr(e)[:300])
            elif cfg.get('relink') is not None:
                st = dict(st, relink_skipped='no_full_graph')
            if cfg.get('edge_link') is not None and fullp is not None and fullp.exists():
                try:
                    import edge_link
                    el = cfg['edge_link']
                    n6, e6, st6 = edge_link.apply(nodes2, ne, fullp, cdir / el.get('model', 'edge_lgb.json'), th=float(el['th']), allow_gap2=bool(el.get('gap2', True)))
                    check_graph(n6, e6, nodes)
                    nodes2, ne = n6, e6; st = dict(st, **st6)
                except Exception as e:
                    st = dict(st, edge_link_error=repr(e)[:300])
            elif cfg.get('edge_link') is not None:
                st = dict(st, edge_link_skipped='no_full_graph')
            if ri_new is not None:
                try:
                    import frag_reinsert
                    n10, e10, st10 = (frag_reinsert.cleanup_both if cfg['frag_reinsert'].get('mode') == 'both' else frag_reinsert.cleanup)(nodes2, ne, ri_new)
                    check_graph(n10, e10, nodes)
                    nodes2, ne = n10, e10; st = dict(st, **st10)
                except Exception as e:
                    nodes2, ne = ri_backup; st = dict(st, ri_cleanup_error=repr(e)[:300])
            if cfg.get('post_prune'):
                try:
                    import prune
                    n8, e8, st8 = prune.prune_fragments(nodes2, ne, int(cfg['post_prune']))
                    check_graph(n8, e8, nodes)
                    nodes2, ne = n8, e8; st = dict(st, post_pruned=st8.get('pruned_nodes', 0))
                except Exception as e:
                    st = dict(st, post_prune_error=repr(e)[:300])
            if cfg.get('term_trim') is not None:
                try:
                    import p14_post
                    tt = cfg['term_trim']
                    n13, e13, st13 = p14_post.term_trim(nodes2, ne, r=float(tt.get('r', 3.5)), minlen=int(tt.get('minlen', 3)), iters=int(tt.get('iters', 5)), join=tt.get('join', 'keep_end'))
                    check_graph(n13, e13, nodes)
                    nodes2, ne = n13, e13; st = dict(st, trimmed=st13.get('trim', 0))
                except Exception as e:
                    st = dict(st, term_trim_error=repr(e)[:300])
            if cfg.get('long_link') is not None and fullp is not None and fullp.exists():
                try:
                    import p14_post
                    ll = cfg['long_link']
                    n14, e14, st14 = p14_post.long_link(nodes2, ne, fullp, fe_min=float(ll.get('fe_min', 0.5)), min_um=float(ll.get('min_um', 14.0)))
                    check_graph(n14, e14, nodes)
                    nodes2, ne = n14, e14; st = dict(st, long_linked=st14.get('ll_added', 0))
                except Exception as e:
                    st = dict(st, long_link_error=repr(e)[:300])
            elif cfg.get('long_link') is not None:
                st = dict(st, long_link_skipped='no_full_graph')
            if cfg.get('tbext') is not None and fullp is not None and fullp.exists():
                try:
                    import p15_post
                    tb = cfg['tbext']
                    n16, e16, st16 = p15_post.tbext(nodes2, ne, fullp, K=int(tb.get('K', 100)), minlen=int(tb.get('minlen', 5)), pmin=float(tb.get('pmin', 0.8)), dup=float(tb.get('dup', 3.5)), join=bool(tb.get('join', True)))
                    check_graph(n16, e16, nodes)
                    nodes2, ne = n16, e16; st = dict(st, **st16)
                except Exception as e:
                    st = dict(st, tbext_error=repr(e)[:300])
            elif cfg.get('tbext') is not None:
                st = dict(st, tbext_skipped='no_full_graph')
            if cfg.get('relinefit') is not None and fullp is not None and fullp.exists():
                try:
                    import p15_post, zarr as _zarr
                    rf = cfg['relinefit']
                    refp = Path(a.graphs).parent / 'reference_graphs' / (name + '.json')
                    shp = tuple(int(x) for x in _zarr.open_group(str(Path(a.data) / (name + '.zarr')), mode='r')['0'].shape[1:])
                    n15, e15, st15 = p15_post.relinefit(nodes2, ne, refp, fullp, shape=shp, w=float(rf.get('w', 0.5)), minsep=float(rf.get('minsep', 2.0)))
                    check_graph(n15, e15, nodes)
                    nodes2, ne = n15, e15; st = dict(st, **st15)
                except Exception as e:
                    st = dict(st, relinefit_error=repr(e)[:300])
            elif cfg.get('relinefit') is not None:
                st = dict(st, relinefit_skipped='no_full_graph')
            if cfg.get('dense_relink') is not None:
                try:
                    import dense_relink
                    dr = cfg['dense_relink']
                    ddir = Path(os.environ.get('DENSE_DIR') or dr.get('dir') or (str(Path(a.full).parent / 'dense') if a.full else '.'))
                    n11, e11, st11 = dense_relink.apply(nodes2, ne, ddir / (name + '.npz'), fullp, Path(a.config).parent / dr.get('model', 'dense_lgb.json'), th=float(dr['th']))
                    check_graph(n11, e11, nodes)
                    nodes2, ne = n11, e11; st = dict(st, **st11)
                    if dr.get('el2') and cfg.get('edge_link') is not None and fullp is not None and fullp.exists():
                        import edge_link, prune
                        el = cfg['edge_link']
                        n12, e12, st12 = edge_link.apply(nodes2, ne, fullp, Path(a.config).parent / el.get('model', 'edge_lgb.json'), th=float(el['th']), allow_gap2=bool(el.get('gap2', True)))
                        n12, e12, _ = prune.prune_fragments(n12, e12, int(cfg.get('post_prune') or 2))
                        check_graph(n12, e12, nodes)
                        nodes2, ne = n12, e12; st = dict(st, el2_links=st12.get('el_gap1', 0) + st12.get('el_gap2', 0))
                except Exception as e:
                    st = dict(st, dense_relink_error=repr(e)[:300])
            if cfg.get('dc_pass2'):
                try:
                    c2 = dc.score_candidates(er, name, nodes2, ne)
                    e7, st7 = dc.apply(nodes2, ne, c2, **cfg['apply'])
                    import dfork, prune
                    e7, st7b = dfork.resolve(nodes2, e7, **cfg.get('dfork', {'K': 2}))
                    n7, e7, st7c = prune.prune_fragments(nodes2, e7, int(cfg.get('prune_min_len') or 2))
                    check_graph(n7, e7, nodes)
                    nodes2, ne = n7, e7; st = dict(st, dc2_added=st7.get('div_added', 0), dc2_stolen=st7.get('div_stolen', 0), dfork2_removed=st7b.get('dfork_removed', 0))
                except Exception as e:
                    st = dict(st, dc2_error=repr(e)[:300])
            pairs = [(int(e['source_id']), int(e['target_id'])) for e in ne]
            assert len(pairs) == len(set(pairs))
            assert max(Counter(t for s, t in pairs).values(), default=0) <= 1
            assert max(Counter(s for s, t in pairs).values(), default=0) <= 2
            assert all(s in nodes2 and t in nodes2 and int(nodes2[t]['t']) == int(nodes2[s]['t']) + 1 for s, t in pairs)
            assert {int(v['t']) for v in nodes2.values()} == {int(v['t']) for v in nodes.values()}
            nodes = nodes2; edges = ne; stats = dict(stats, **st, candidates=len(cands), seconds=round(time.time() - t0, 2))
        except Exception as e:
            stats = {'error': repr(e)[:300], 'traceback': traceback.format_exc()[-1500:]}
        tmp = dst.with_name(name + '.%d.tmp' % os.getpid())
        tmp.write_text(json.dumps({'nodes': {str(k): v for k, v in nodes.items()}, 'edges': edges, 'pstage': stats}))
        os.replace(tmp, dst)
        print('PSTAGE_MOVIE', name, json.dumps(stats)[:300], flush=True)
        for p in (Path(a.work) / ('pstage_cache_%d' % a.worker)).rglob('*'):
            if p.is_file() and name in p.name: p.unlink()

def export(a, started):
    import zarr
    data = Path(a.data); names = sorted(p.stem for p in data.glob('*.zarr'))
    cols = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
    output = Path(a.submission); tmp = output.with_name('submission.pstage.tmp'); row_id = 0; records = []; clipped = 0
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for name in names:
            d = json.loads((Path(a.out) / (name + '.json')).read_text())
            nodes = {int(k): v for k, v in d['nodes'].items()}; edges = d['edges']
            shape = zarr.open_group(str(data / (name + '.zarr')), mode='r')['0'].shape
            assert {int(v['t']) for v in nodes.values()} == set(range(shape[0])), name + ' missing frames'
            for n in sorted(nodes):
                v = nodes[n]; c0 = [max(0, int(round(v[k]))) for k in ['z', 'y', 'x']]
                c = [min(x, lim - 1) for x, lim in zip(c0, shape[1:])]
                if c != c0: clipped += 1
                w.writerow(dict(id=row_id, dataset=name, row_type='node', node_id=n, t=int(v['t']), z=c[0], y=c[1], x=c[2], source_id=-1, target_id=-1)); row_id += 1
            pairs = [(int(e['source_id']), int(e['target_id'])) for e in edges]
            assert len(pairs) == len(set(pairs)); assert max(Counter(s for s, t in pairs).values(), default=0) <= 2; assert max(Counter(t for s, t in pairs).values(), default=0) <= 1
            for s, t in pairs:
                assert s in nodes and t in nodes and nodes[t]['t'] == nodes[s]['t'] + 1
                w.writerow(dict(id=row_id, dataset=name, row_type='edge', node_id=-1, t=-1, z=-1, y=-1, x=-1, source_id=s, target_id=t)); row_id += 1
            records.append({'movie': name, 'nodes': len(nodes), 'edges': len(edges), 'pstage': d.get('pstage', {})})
    os.replace(tmp, output)
    rep = {'movies': len(names), 'rows': row_id, 'seconds': time.time() - started, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'records': records,
           'errors': sum(1 for r in records if 'error' in r['pstage']), 'clipped_nodes': clipped, 'forks_added': sum(r['pstage'].get('div_added', 0) for r in records), 'dsr_added': sum(r['pstage'].get('dsr_added', 0) for r in records), 'dfork_removed': sum(r['pstage'].get('dfork_removed', 0) for r in records), 'dsr_errors': sum(1 for r in records if 'dsr_error' in r['pstage'] or 'dfork_error' in r['pstage']), 'relinked': sum(r['pstage'].get('rl_applied', 0) for r in records), 'edge_linked': sum(r['pstage'].get('el_gap1', 0) + r['pstage'].get('el_gap2', 0) for r in records), 'link_errors': sum(1 for r in records if 'relink_error' in r['pstage'] or 'edge_link_error' in r['pstage']), 'trimmed': sum(r['pstage'].get('trimmed', 0) for r in records), 'long_linked': sum(r['pstage'].get('long_linked', 0) for r in records), 'p14_errors': sum(1 for r in records if 'term_trim_error' in r['pstage'] or 'long_link_error' in r['pstage']), 'relinefit_moved': sum(r['pstage'].get('rlf_moved', 0) for r in records), 'tb_added': sum(r['pstage'].get('tb_add_start', 0) + r['pstage'].get('tb_add_end', 0) for r in records), 'tb_joined': sum(r['pstage'].get('tb_join', 0) for r in records), 'p15_errors': sum(1 for r in records if any(k in r['pstage'] for k in ('relinefit_error', 'relinefit_skipped', 'tbext_error', 'tbext_skipped')))}
    Path(a.work, 'pstage_report.json').write_text(json.dumps(rep, indent=1))
    print('PSTAGE_EXPORT_COMPLETE', json.dumps({k: v for k, v in rep.items() if k != 'records'}), flush=True)

def main(a):
    import torch
    started = time.time(); work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in Path(a.data).glob('*.zarr'))
    ngpu = int(os.environ.get('DN_NW', '2'))
    shards = [[] for _ in range(ngpu)]; sizes = [0] * ngpu
    for n in sorted(names, key=lambda n: (Path(a.graphs) / (n + '.json')).stat().st_size, reverse=True):
        j = min(range(ngpu), key=lambda i: sizes[i]); shards[j].append(n); sizes[j] += (Path(a.graphs) / (n + '.json')).stat().st_size
    procs = []
    for g, ns in enumerate(shards):
        (work / ('pstage_shard_%d.json' % g)).write_text(json.dumps(ns))
        cmd = [sys.executable, '-u', __file__, '--worker', str(g)] + sum([['--' + k, str(getattr(a, k))] for k in ['artifact', 'config', 'data', 'graphs', 'out', 'work', 'submission', 'deadline', 'full']], [])
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(g % 2), 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        procs.append(subprocess.Popen(cmd, env=env, stdout=(work / ('pstage_worker_%d.log' % g)).open('w'), stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in procs):
        done = len(list(Path(a.out).glob('*.json'))) if Path(a.out).exists() else 0
        print('PSTAGE_PROGRESS', done, '/', len(names), round((time.time() - started) / 60, 1), 'min', flush=True); time.sleep(15)
    for g, p in enumerate(procs):
        if p.returncode != 0: raise RuntimeError('pstage worker failed: ' + (work / ('pstage_worker_%d.log' % g)).read_text()[-4000:])
    export(a, started)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--worker', type=int)
    for k in ['artifact', 'config', 'data', 'graphs', 'out', 'work', 'submission']: p.add_argument('--' + k, required=True)
    p.add_argument('--deadline', default='0')
    p.add_argument('--full', default='')
    a = p.parse_args()
    worker(a) if a.worker is not None else main(a)
