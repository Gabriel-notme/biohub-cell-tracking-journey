import json, sys, copy
from pathlib import Path
# usage: build_kernel.py <base_nb> <variant_name> <slug> <title> <config_file> <base_label> <out_dir> <base_meta_json>
base_nb, variant, slug, title, cfg_file, base_label, out_dir, base_meta = sys.argv[1:9]
nb = json.load(open(base_nb))
cells = nb['cells']
md = {'cell_type': 'markdown', 'metadata': {}, 'source': [
    '# Biohub %s: %s base + division completion\n\n' % (variant, base_label),
    'Runs the frozen %s pipeline unchanged, then applies a post-lineage **division completion** stage: ' % base_label,
    'the frozen image-conditioned event model (b1 fork head) re-scores every candidate fork on the final lineage graph ',
    '(parent with one child + a second daughter that either starts a new track or is held by another parent), ',
    'adds high-confidence forks, and prunes isolated single-node components. Every movie falls back to its unmodified graph on error, ',
    'and the whole stage falls back to the %s submission if validation fails or the time budget is exhausted.\n\n' % base_label,
    'Local validation (official metric, not a leaderboard score): 36 held-out movies B5 0.9693 -> 0.9749; 32 audit movies 0.9543 -> 0.9724 (audit set overlaps b1 training, optimistic).']}
cells[0] = md
code = r'''
import shutil, subprocess, json, hashlib, time, os, sys
from pathlib import Path
import pandas as pd
P_SLUG = 'biohub-p12-division-stage'
P_CONFIG = '__CFG__'
p_roots = [Path('/kaggle/input/datasets/shawsebastian') / P_SLUG, Path('/kaggle/input') / P_SLUG]
P_ROOT = next((p for p in p_roots if (p / 'p_stage.py').is_file()), None)
work = Path('/kaggle/working')
base_sub = work / 'submission.csv'; backup = work / 'submission_base.csv'
shutil.copyfile(base_sub, backup)
elapsed_h = (time.monotonic() - NOTEBOOK_STARTED) / 3600
status = {'variant': '__VARIANT__', 'elapsed_hours_before_stage': round(elapsed_h, 3), 'applied': False}

def validate_submission(path):
    df = pd.read_csv(path)
    cols = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
    assert df.columns.tolist() == cols
    assert df['id'].tolist() == list(range(len(df)))
    expected = sorted(p.name.removesuffix('.zarr') for p in TEST_DIR.iterdir() if p.name.endswith('.zarr'))
    assert sorted(df['dataset'].astype(str).unique()) == expected
    for ds, g in df.groupby('dataset'):
        n = g[g.row_type.eq('node')]; e = g[g.row_type.eq('edge')]
        assert len(n) > 0 and n.node_id.is_unique and (n[['t', 'z', 'y', 'x']] >= 0).all().all()
        tt = dict(zip(n.node_id.astype(int), n.t.astype(int)))
        assert all(int(s) in tt and int(d) in tt and tt[int(s)] + 1 == tt[int(d)] for s, d in zip(e.source_id, e.target_id))
        assert e.target_id.value_counts().max() <= 1 if len(e) else True
        assert e.source_id.value_counts().max() <= 2 if len(e) else True
        assert not e.duplicated(['source_id', 'target_id']).any()
    return len(df)

if P_ROOT is None:
    status['note'] = 'stage dataset missing'
elif elapsed_h > 10.0:
    status['note'] = 'skipped: time budget'
else:
    deadline = time.time() + max(0.0, 11.0 - elapsed_h) * 3600
    cmd = [sys.executable, '-u', str(P_ROOT / 'p_stage.py'), '--artifact', str(LINEAGE_ROOT), '--config', str(P_ROOT / P_CONFIG),
           '--data', str(TEST_DIR), '--graphs', str(work / 'lineage_graphs'), '--out', str(work / 'pstage_graphs'),
           '--work', str(work / 'pstage'), '--submission', str(base_sub), '--deadline', str(deadline)]
    try:
        r = subprocess.run(cmd, env={**os.environ, 'BIOHUB_BASE_REPO': str(REPO_DIR)}, timeout=max(600, deadline - time.time() + 1200))
        status['returncode'] = r.returncode
        if r.returncode == 0:
            status['rows'] = validate_submission(base_sub); status['applied'] = True
    except Exception as e:
        status['note'] = repr(e)[:500]
    if not status['applied']:
        shutil.copyfile(backup, base_sub)
        status['rows'] = validate_submission(base_sub)
rep = work / 'pstage' / 'pstage_report.json'
if rep.exists():
    r = json.loads(rep.read_text()); status.update({k: r[k] for k in ['movies', 'errors', 'forks_added', 'seconds'] if k in r})
status['final_sha256'] = hashlib.sha256(base_sub.read_bytes()).hexdigest()
status['total_notebook_hours'] = round((time.monotonic() - NOTEBOOK_STARTED) / 3600, 3)
(work / 'pstage_status.json').write_text(json.dumps(status, indent=2))
print('PSTAGE_STATUS', json.dumps(status))
'''.replace('__CFG__', cfg_file).replace('__VARIANT__', variant)
cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [code]})
for c in cells:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
nb['cells'] = cells
out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
json.dump(nb, open(out / 'model.ipynb', 'w'), indent=1)
meta = json.load(open(base_meta))
meta.pop('id_no', None); meta['id'] = 'shawsebastian/' + slug; meta['title'] = title; meta['code_file'] = 'model.ipynb'
if 'shawsebastian/biohub-p12-division-stage' not in meta['dataset_sources']: meta['dataset_sources'].append('shawsebastian/biohub-p12-division-stage')
meta['is_private'] = True; meta['enable_internet'] = False; meta['enable_gpu'] = True
json.dump(meta, open(out / 'kernel-metadata.json', 'w'), indent=2)
print('built', out, len(cells), 'cells', meta)
