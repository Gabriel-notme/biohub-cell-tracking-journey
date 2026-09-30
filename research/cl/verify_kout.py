#!/usr/bin/env python3
"""Verify a downloaded P-series kernel output directory before submitting it.

usage: python3 verify_kout.py <kout_dir> --variant P5 [--test-dir /workspace/data/test]
                              [--dataset-dir /workspace/p12ds] [--allow-errors]
Checks submission.csv structure (same rules as the notebook's validate_submission),
pstage_status.json (applied, returncode 0, errors 0, rows/sha match the CSV),
pstage/pstage_report.json (errors 0, rows/sha match), and the kernel log
(FULL_PIPELINE_VALIDATED + PSTAGE_STATUS present, runtime). Read-only; prints VERIFY_OK / VERIFY_FAIL.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('kout'); ap.add_argument('--variant', required=True)
ap.add_argument('--test-dir', default='/workspace/data/test')
ap.add_argument('--dataset-dir', default='/workspace/p12ds')
ap.add_argument('--allow-errors', action='store_true', help='do not fail on per-movie P-stage errors (they fall back to the base graph)')
a = ap.parse_args()
K = Path(a.kout); fails = []; info = {}

def need(cond, msg):
    if not cond: fails.append(msg)

# ---- submission.csv ----------------------------------------------------------
sub = K / 'submission.csv'
need(sub.is_file(), 'submission.csv missing')
if sub.is_file():
    raw = sub.read_bytes(); sha = hashlib.sha256(raw).hexdigest(); info['csv_sha256'] = sha
    df = pd.read_csv(sub); info['csv_rows'] = len(df)
    cols = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
    need(df.columns.tolist() == cols, 'bad columns %s' % df.columns.tolist())
    need(df['id'].tolist() == list(range(len(df))), 'id not 0..n-1')
    expected = sorted(p.name[:-5] for p in Path(a.test_dir).iterdir() if p.name.endswith('.zarr'))
    need(sorted(df['dataset'].astype(str).unique()) == expected, 'dataset set mismatch vs %s' % a.test_dir)
    per = {}
    for ds, g in df.groupby('dataset'):
        n = g[g.row_type.eq('node')]; e = g[g.row_type.eq('edge')]
        tt = dict(zip(n.node_id.astype(int), n.t.astype(int)))
        ok = (len(n) > 0 and n.node_id.is_unique and bool((n[['t', 'z', 'y', 'x']] >= 0).all().all())
              and all(int(s) in tt and int(d) in tt and tt[int(s)] + 1 == tt[int(d)] for s, d in zip(e.source_id, e.target_id))
              and (e.target_id.value_counts().max() <= 1 if len(e) else True)
              and (e.source_id.value_counts().max() <= 2 if len(e) else True)
              and not e.duplicated(['source_id', 'target_id']).any())
        need(ok, 'structural check failed for %s' % ds)
        per[ds] = {'nodes': len(n), 'edges': len(e), 'forks': int((e.source_id.value_counts() == 2).sum()) if len(e) else 0}
    info['per_movie'] = per
base = K / 'submission_base.csv'
if base.is_file():
    info['base_rows'] = sum(1 for _ in base.open()) - 1
    info['base_sha256'] = hashlib.sha256(base.read_bytes()).hexdigest()
    need(info['base_sha256'] != info.get('csv_sha256'), 'submission.csv identical to submission_base.csv (P-stage not applied?)')

# ---- pstage_status.json --------------------------------------------------------
st_p = K / 'pstage_status.json'
need(st_p.is_file(), 'pstage_status.json missing')
if st_p.is_file():
    st = json.loads(st_p.read_text()); info['status'] = st
    need(st.get('variant') == a.variant, 'variant %r != %r' % (st.get('variant'), a.variant))
    need(st.get('applied') is True, 'applied is not true (note=%r)' % st.get('note'))
    need(st.get('returncode') == 0, 'returncode %r' % st.get('returncode'))
    need(st.get('rows') == info.get('csv_rows'), 'status rows %r != csv rows %r' % (st.get('rows'), info.get('csv_rows')))
    need(st.get('final_sha256') == info.get('csv_sha256'), 'status final_sha256 != csv sha256')
    if not a.allow_errors:
        need(st.get('errors', 0) == 0, 'status errors=%r' % st.get('errors'))
        need(st.get('dsr_errors', 0) == 0, 'status dsr_errors=%r' % st.get('dsr_errors'))
    need(st.get('total_notebook_hours', 99) < 11.0, 'total_notebook_hours %r' % st.get('total_notebook_hours'))
    hashes = st.get('stage_files_sha256')
    if isinstance(hashes, dict) and hashes:
        local = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in Path(a.dataset_dir).glob('*') if p.is_file()}
        mism = sorted(k for k in hashes if local.get(k) != hashes[k])
        info['stage_files_mismatch_vs_local'] = mism
        need(not mism, 'mounted dataset files differ from local %s: %s (old dataset version mounted?)' % (a.dataset_dir, mism))

# ---- pstage_report.json --------------------------------------------------------------
rp = K / 'pstage' / 'pstage_report.json'
if rp.is_file():
    r = json.loads(rp.read_text())
    info['report'] = {k: v for k, v in r.items() if k != 'records'}
    need(r.get('rows') == info.get('csv_rows'), 'report rows %r != csv rows' % r.get('rows'))
    need(r.get('sha256') == info.get('csv_sha256'), 'report sha256 != csv sha256')
    need(r.get('movies') == len(info.get('per_movie', {})), 'report movies mismatch')
    if not a.allow_errors:
        need(r.get('errors', 0) == 0, 'report errors=%r' % r.get('errors'))
        bad = [x['movie'] for x in r.get('records', []) if 'error' in x.get('pstage', {}) or 'skipped' in x.get('pstage', {})]
        need(not bad, 'movies with error/skip: %s' % bad)
else:
    print('NOTE: pstage/pstage_report.json not downloaded (use full download or include pstage/ in --file-pattern)')

# ---- kernel log ----------------------------------------------------------------
logs = [p for p in K.glob('*.log') if p.name.startswith('biohub-')]
if logs:
    try:
        ev = json.loads(logs[0].read_text())
        text = ''.join(e.get('data', '') for e in ev)
        info['kernel_log'] = {'file': logs[0].name, 'runtime_min': round(ev[-1]['time'] / 60, 1),
                              'FULL_PIPELINE_VALIDATED': 'FULL_PIPELINE_VALIDATED' in text, 'PSTAGE_STATUS': 'PSTAGE_STATUS' in text,
                              'Traceback_count': text.count('Traceback (most recent call last)')}
        need('FULL_PIPELINE_VALIDATED' in text, 'FULL_PIPELINE_VALIDATED missing from kernel log')
        need('PSTAGE_STATUS' in text, 'PSTAGE_STATUS missing from kernel log')
    except Exception as e:
        print('NOTE: could not parse kernel log', logs[0], repr(e)[:200])
else:
    print('NOTE: kernel log not found in', K)

print(json.dumps(info, indent=1, default=str))
print('VERIFY_FAIL' if fails else 'VERIFY_OK', json.dumps(fails))
sys.exit(1 if fails else 0)

