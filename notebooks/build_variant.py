"""Build a new P-stage notebook variant from a verified one. Only three things change: the P-stage config file name, the
variant label written to pstage_status.json, and the markdown description cell. The script refuses to write anything if
any other line would differ, or if the config uses a key that the notebook's stage script does not implement (stage scripts
silently ignore unknown keys, e.g. p_stage12.py has no 'p22jump'; the P7 notebook's p_stage4.py has no P14+ steps).

usage (from the repository root):
  python notebooks/build_variant.py <source notebook dir> <config file name> <variant> <kernel slug> <title> <description> [owner]
  e.g. python notebooks/build_variant.py notebooks/p21-final-selection p23c_config.json P23C my-p23c "My P23C" "P21, start 0.80" me
Writes notebooks/<kernel slug>/model.ipynb and kernel-metadata.json (the folder must not exist yet). dataset_sources in
kernel-metadata.json still name the original datasets; edit them if you use your own copies."""
import json, re, sys
from pathlib import Path

src_dir, cfg, variant, slug, title, desc = sys.argv[1:7]
owner = sys.argv[7] if len(sys.argv) > 7 else None
assert re.fullmatch(r'[A-Za-z0-9_.-]+\.json', cfg), 'config must be a plain *.json file name in pstage/'
assert re.fullmatch(r'[A-Za-z0-9_.-]+', variant), 'variant label: letters, digits, _ . - only'
assert re.fullmatch(r'[a-z0-9][a-z0-9-]*', slug), 'kernel slug: lower-case letters, digits and - only'
src_dir = Path(src_dir)
raw = (src_dir / 'model.ipynb').read_bytes()
newline = '\r\n' if b'\r\n' in raw else '\n'
nb = json.loads(raw.decode('utf-8'))
meta = json.loads((src_dir / 'kernel-metadata.json').read_text(encoding='utf-8'))
old = [''.join(c['source']) for c in nb['cells']]

text = '\n'.join(old)
cur_cfg = re.search(r"P_CONFIG = '([^']+)'", text).group(1)
cur_var = re.search(r"status = \{'variant': '([^']+)'", text).group(1)

# the config must only use keys implemented by the stage script this notebook runs
pstage = Path(__file__).resolve().parent.parent / 'pstage'
stage = sorted(set(re.findall(r"p_stage\d*\.py", text)))
assert len(stage) == 1, stage
stage_src = (pstage / stage[0]).read_text(encoding='utf-8')
conf = json.loads((pstage / cfg).read_text(encoding='utf-8'))
keys = [k for k in conf if k != 'variant']
missing = [k for k in keys if "'%s'" % k not in stage_src]
assert not missing, '%s does not implement config keys %s; use a notebook that runs a later stage script' % (stage[0], missing)
# model files named by the config must ship with the P-stage dataset (DivNet weights are not in this repository)
models = [conf[k]['model'] for k in conf if isinstance(conf[k], dict) and 'model' in conf[k]] + list(conf.get('divnet', {}).get('models', []))
absent = [m for m in models if not (pstage / m).is_file()]
assert not absent, 'model files missing from pstage/: %s (without them the notebook would silently submit the B5 base)' % absent

n_cfg = n_var = n_desc = 0
for c in nb['cells']:
    s = ''.join(c['source'])
    if c['cell_type'] == 'markdown' and n_desc == 0:
        c['source'] = [desc]; n_desc += 1
        continue
    if "P_CONFIG = '%s'" % cur_cfg in s:
        n_cfg += 1; s = s.replace("P_CONFIG = '%s'" % cur_cfg, "P_CONFIG = '%s'" % cfg)
    if "'variant': '%s'" % cur_var in s:
        n_var += 1; s = s.replace("'variant': '%s'" % cur_var, "'variant': '%s'" % variant)
    c['source'] = s.splitlines(keepends=True)
assert (n_cfg, n_var, n_desc) == (1, 1, 1), (n_cfg, n_var, n_desc)

new = [''.join(c['source']) for c in nb['cells']]
changed = [(i, a, b) for i, (x, y) in enumerate(zip(old, new)) for a, b in zip(x.splitlines(), y.splitlines()) if a != b]
assert all(("P_CONFIG" in a) or ("'variant'" in a) or nb['cells'][i]['cell_type'] == 'markdown' for i, a, b in changed), changed

out = Path(__file__).resolve().parent / slug
assert not out.exists(), 'output folder exists (refusing to overwrite): %s' % out
out.mkdir(parents=True)
(out / 'model.ipynb').write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8', newline=newline)
meta['id'] = '%s/%s' % (owner or meta['id'].split('/')[0], slug)
meta['title'] = title
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8', newline=newline)
print('wrote', out, '| stage', stage[0], '| config', cur_cfg, '->', cfg, '| variant', cur_var, '->', variant)
