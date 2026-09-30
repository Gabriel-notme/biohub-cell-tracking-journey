#!/usr/bin/env python3
"""Build a P-series Kaggle kernel dir from the pushed P3 kernel (shawsebastian/biohub-p3-sister-recovery v2).

The output notebook is P3 with only these changes:
  * cell 0 (markdown)            -> <description>
  * last cell (P-stage):  P_CONFIG = '<config>'  and  'variant': '<variant>'
  * optional (flags):     stage script name (default p_stage2.py, unchanged),
                          extra keys copied from pstage_report.json into pstage_status.json,
                          sha256 of the mounted stage files recorded in pstage_status.json.
kernel-metadata.json = P3 metadata with new id/title (private, GPU on, internet off,
same dataset/competition sources, docker image and machine shape).

usage:
  python3 build_pkernel.py <variant> <slug> <title> <config_file> <out_dir> <description>
         [--stage-script p_stage2.py] [--extra-report-keys k1,k2] [--record-hashes]
         [--base-dir /workspace/kernels/p3] [--dataset-dir /workspace/p12ds] [--force]

This script only writes files under <out_dir>. It never calls the Kaggle API.
"""
import argparse, ast, difflib, hashlib, json, re, sys
from pathlib import Path

OWNER = 'shawsebastian'
P_DATASET = 'shawsebastian/biohub-p12-division-stage'
REPORT_KEYS_P3 = "['movies', 'errors', 'forks_added', 'dsr_added', 'dfork_removed', 'dsr_errors', 'seconds']"


def slugify(title):
    return re.sub(r'-+', '-', re.sub(r'[^a-z0-9]+', '-', title.lower())).strip('-')


def sub_once(s, old, new, what):
    n = s.count(old)
    if n != 1:
        sys.exit('ERROR: expected exactly 1 occurrence of %s in the P-stage cell, found %d' % (what, n))
    return s.replace(old, new)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('variant'); ap.add_argument('slug'); ap.add_argument('title')
    ap.add_argument('config_file'); ap.add_argument('out_dir'); ap.add_argument('description')
    ap.add_argument('--stage-script', default='p_stage2.py')
    ap.add_argument('--extra-report-keys', default='')
    ap.add_argument('--record-hashes', action='store_true')
    ap.add_argument('--base-dir', default='/workspace/kernels/p3')
    ap.add_argument('--dataset-dir', default='/workspace/p12ds')
    ap.add_argument('--force', action='store_true', help='overwrite an existing out_dir')
    a = ap.parse_args()

    # ---- argument checks -------------------------------------------------
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', a.variant): sys.exit('ERROR: bad variant name')
    if not re.fullmatch(r'[a-z0-9]+(-[a-z0-9]+)*', a.slug): sys.exit('ERROR: slug must be lowercase letters/digits/hyphens')
    if a.slug.startswith(OWNER + '/'): sys.exit('ERROR: pass the slug without the owner prefix')
    if not 5 <= len(a.title) <= 50: sys.exit('ERROR: Kaggle kernel titles must be 5-50 characters')
    if slugify(a.title) != a.slug:
        sys.exit('ERROR: title %r slugifies to %r, not %r (Kaggle warns and may create a different slug)' % (a.title, slugify(a.title), a.slug))
    if not re.fullmatch(r'[A-Za-z0-9_.-]+\.json', a.config_file): sys.exit('ERROR: config must be a bare *.json file name inside the dataset')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+\.py', a.stage_script): sys.exit('ERROR: stage script must be a bare *.py file name')
    extra_keys = [k.strip() for k in a.extra_report_keys.split(',') if k.strip()]
    if any(not re.fullmatch(r'[A-Za-z0-9_]+', k) for k in extra_keys): sys.exit('ERROR: bad report key')
    base = Path(a.base_dir); out = Path(a.out_dir).resolve(); ds_root = Path(a.dataset_dir).resolve()
    if out == ds_root or ds_root in out.parents: sys.exit('ERROR: never build a kernel inside the dataset folder (it would be uploaded)')
    if out == base.resolve(): sys.exit('ERROR: refusing to overwrite the base kernel dir')
    if out.exists() and any(out.iterdir()) and not a.force: sys.exit('ERROR: %s exists and is not empty (use --force)' % out)
    ds = Path(a.dataset_dir)
    warnings = []
    for f in (a.config_file, a.stage_script):
        if not (ds / f).is_file(): warnings.append('%s not found in %s - add it before `kaggle datasets version`' % (f, ds))
    if (ds / a.config_file).is_file():
        try: json.loads((ds / a.config_file).read_text())
        except Exception as e: sys.exit('ERROR: %s is not valid JSON: %r' % (ds / a.config_file, e))

    # ---- notebook ------------------------------------------------------------
    src_nb = json.loads((base / 'model.ipynb').read_text())
    nb = json.loads(json.dumps(src_nb))
    cells = nb['cells']
    assert cells[0]['cell_type'] == 'markdown' and cells[-1]['cell_type'] == 'code'
    s = ''.join(cells[-1]['source'])
    assert "P_SLUG = 'biohub-p12-division-stage'" in s and "'--full', str(work / 'fullgraphs')" in s, 'unexpected P-stage cell'
    s = re.sub(r"P_CONFIG = '[^']*'", lambda m: "P_CONFIG = '%s'" % a.config_file, s, count=1)
    s = re.sub(r"status = \{'variant': '[^']*'", lambda m: "status = {'variant': '%s'" % a.variant, s, count=1)
    if a.stage_script != 'p_stage2.py':
        s = sub_once(s, "(p / 'p_stage2.py').is_file()", "(p / '%s').is_file()" % a.stage_script, 'P_ROOT probe')
        s = sub_once(s, "str(P_ROOT / 'p_stage2.py')", "str(P_ROOT / '%s')" % a.stage_script, 'stage command')
    if extra_keys:
        p3_keys = ast.literal_eval(REPORT_KEYS_P3)
        keys = p3_keys + [k for k in extra_keys if k not in p3_keys]
        s = sub_once(s, REPORT_KEYS_P3, repr(keys), 'report key list')
    if a.record_hashes:
        anchor = "status['final_sha256'] = hashlib.sha256(base_sub.read_bytes()).hexdigest()\n"
        hook = ("try:\n"
                "    status['stage_files_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(P_ROOT.glob('*')) if p.is_file()} if P_ROOT else {}\n"
                "except Exception as e:\n"
                "    status['stage_files_sha256'] = repr(e)[:200]\n")
        s = sub_once(s, anchor, hook + anchor, 'final_sha256 anchor')
    assert "P_CONFIG = '%s'" % a.config_file in s and "'variant': '%s'" % a.variant in s
    compile(s, 'pstage_cell', 'exec')  # syntax check of the edited cell
    cells[-1]['source'] = [s]
    cells[0] = {'cell_type': 'markdown', 'metadata': {}, 'source': [a.description]}
    for c in cells:
        if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None

    # ---- metadata ------------------------------------------------------------
    meta = json.loads((base / 'kernel-metadata.json').read_text())
    meta.pop('id_no', None)
    meta['id'] = OWNER + '/' + a.slug; meta['title'] = a.title; meta['code_file'] = 'model.ipynb'
    meta['is_private'] = True; meta['enable_gpu'] = True; meta['enable_tpu'] = False; meta['enable_internet'] = False
    assert P_DATASET in meta['dataset_sources'], 'P-stage dataset missing from P3 metadata'
    assert meta['competition_sources'] == ['biohub-cell-tracking-during-development']

    out.mkdir(parents=True, exist_ok=True)
    (out / 'model.ipynb').write_text(json.dumps(nb, indent=1))
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))

    # ---- verification / diff summary ----------------------------------------------
    chk = json.loads((out / 'model.ipynb').read_text())
    json.loads((out / 'kernel-metadata.json').read_text())
    assert len(chk['cells']) == len(src_nb['cells'])
    print('BUILT', out)
    print('  notebook sha256', hashlib.sha256((out / 'model.ipynb').read_bytes()).hexdigest(), 'cells', len(chk['cells']))
    for i, (x, y) in enumerate(zip(src_nb['cells'], chk['cells'])):
        sx, sy = ''.join(x['source']), ''.join(y['source'])
        tag = 'same' if (sx == sy and x['cell_type'] == y['cell_type']) else 'CHANGED'
        print('  cell %d %-8s %-7s %s' % (i, y['cell_type'], tag, hashlib.sha256(sy.encode()).hexdigest()[:12]))
        if tag == 'CHANGED' and i > 0:
            for ln in difflib.unified_diff(sx.splitlines(), sy.splitlines(), 'p3', a.variant, n=0, lineterm=''):
                if not ln.startswith(('---', '+++', '@@')): print('      ' + ln[:200])
    unexpected = [i for i, (x, y) in enumerate(zip(src_nb['cells'], chk['cells']))
                  if 0 < i < len(chk['cells']) - 1 and ''.join(x['source']) != ''.join(y['source'])]
    assert not unexpected, 'unexpected changes in cells %s' % unexpected
    base_meta = json.loads((base / 'kernel-metadata.json').read_text())
    for k in sorted(set(base_meta) | set(meta)):
        if base_meta.get(k) != meta.get(k): print('  meta %-16s %r -> %r' % (k, base_meta.get(k), meta.get(k)))
    for w in warnings: print('WARNING:', w)
    print('  next: kaggle kernels push -p %s' % out)


if __name__ == '__main__':
    main()

