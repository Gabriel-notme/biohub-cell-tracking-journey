"""P-series post-lineage stage: division completion on the final lineage graph.

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

def worker(a):
    os.environ['BIOHUB_ART'] = a.artifact
    import numpy as np, torch
    sys.path.insert(0, a.artifact); sys.path.insert(0, str(Path(__file__).parent))
    from refine_events import EventRefiner
    import div_complete as dc
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
            cands = dc.score_candidates(er, name, nodes, edges)
            ne, st = dc.apply(nodes, edges, cands, **cfg['apply'])
            if cfg.get('prune_min_len'):
                import prune
                nodes2, ne, pst = prune.prune_fragments(nodes, ne, int(cfg['prune_min_len'])); st = dict(st, **pst)
            else:
                nodes2 = nodes
            pairs = [(int(e['source_id']), int(e['target_id'])) for e in ne]
            assert len(pairs) == len(set(pairs))
            assert max(Counter(t for s, t in pairs).values(), default=0) <= 1
            assert max(Counter(s for s, t in pairs).values(), default=0) <= 2
            assert all(s in nodes2 and t in nodes2 and int(nodes2[t]['t']) == int(nodes2[s]['t']) + 1 for s, t in pairs)
            assert {int(v['t']) for v in nodes2.values()} == {int(v['t']) for v in nodes.values()}
            nodes = nodes2; edges = ne; stats = dict(st, candidates=len(cands), seconds=round(time.time() - t0, 2))
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
    output = Path(a.submission); tmp = output.with_name('submission.pstage.tmp'); row_id = 0; records = []
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for name in names:
            d = json.loads((Path(a.out) / (name + '.json')).read_text())
            nodes = {int(k): v for k, v in d['nodes'].items()}; edges = d['edges']
            shape = zarr.open_group(str(data / (name + '.zarr')), mode='r')['0'].shape
            assert {int(v['t']) for v in nodes.values()} == set(range(shape[0])), name + ' missing frames'
            for n in sorted(nodes):
                v = nodes[n]; c = [max(0, int(round(v[k]))) for k in ['z', 'y', 'x']]
                assert all(x < lim for x, lim in zip(c, shape[1:])), (name, 'coordinate outside image')
                w.writerow(dict(id=row_id, dataset=name, row_type='node', node_id=n, t=int(v['t']), z=c[0], y=c[1], x=c[2], source_id=-1, target_id=-1)); row_id += 1
            pairs = [(int(e['source_id']), int(e['target_id'])) for e in edges]
            assert len(pairs) == len(set(pairs)); assert max(Counter(s for s, t in pairs).values(), default=0) <= 2; assert max(Counter(t for s, t in pairs).values(), default=0) <= 1
            for s, t in pairs:
                assert s in nodes and t in nodes and nodes[t]['t'] == nodes[s]['t'] + 1
                w.writerow(dict(id=row_id, dataset=name, row_type='edge', node_id=-1, t=-1, z=-1, y=-1, x=-1, source_id=s, target_id=t)); row_id += 1
            records.append({'movie': name, 'nodes': len(nodes), 'edges': len(edges), 'pstage': d.get('pstage', {})})
    os.replace(tmp, output)
    rep = {'movies': len(names), 'rows': row_id, 'seconds': time.time() - started, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'records': records,
           'errors': sum(1 for r in records if 'error' in r['pstage']), 'forks_added': sum(r['pstage'].get('div_added', 0) for r in records)}
    Path(a.work, 'pstage_report.json').write_text(json.dumps(rep, indent=1))
    print('PSTAGE_EXPORT_COMPLETE', json.dumps({k: v for k, v in rep.items() if k != 'records'}), flush=True)

def main(a):
    import torch
    started = time.time(); work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in Path(a.data).glob('*.zarr'))
    ngpu = max(1, min(2, torch.cuda.device_count()))
    shards = [[] for _ in range(ngpu)]; sizes = [0] * ngpu
    for n in sorted(names, key=lambda n: (Path(a.graphs) / (n + '.json')).stat().st_size, reverse=True):
        j = min(range(ngpu), key=lambda i: sizes[i]); shards[j].append(n); sizes[j] += (Path(a.graphs) / (n + '.json')).stat().st_size
    procs = []
    for g, ns in enumerate(shards):
        (work / ('pstage_shard_%d.json' % g)).write_text(json.dumps(ns))
        cmd = [sys.executable, '-u', __file__, '--worker', str(g)] + sum([['--' + k, str(getattr(a, k))] for k in ['artifact', 'config', 'data', 'graphs', 'out', 'work', 'submission', 'deadline']], [])
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(g), 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
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
    a = p.parse_args()
    worker(a) if a.worker is not None else main(a)
