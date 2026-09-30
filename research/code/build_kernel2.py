import json, sys, re
from pathlib import Path
# usage: build_kernel2.py <p1_kernel_nb> <variant> <slug> <title> <config_file> <out_dir> <base_meta_json> <desc>
src_nb, variant, slug, title, cfg_file, out_dir, base_meta, desc = sys.argv[1:9]
nb = json.load(open(src_nb)); cells = nb['cells']
# 1) cell with the predict-script patch: save the pre-ILP candidate graph (no effect on B5 outputs)
PATCH = r'''
os.environ['FULLGRAPH_DIR'] = '/kaggle/working/fullgraphs'
_fg_src = _batch_path.read_text()
_fg_old = "graph = build_graph(coords, edges)\n"
_fg_new = "graph = build_graph(coords, edges)\n        if os.environ.get('FULLGRAPH_DIR'):\n            _fgd = Path(os.environ['FULLGRAPH_DIR']); _fgd.mkdir(parents=True, exist_ok=True)\n            save_graph(graph, _fgd / f'{name}.geff')\n"
assert _fg_src.count(_fg_old) >= 1, 'fullgraph anchor'
_batch_path.write_text(_fg_src.replace(_fg_old, _fg_new))
print('FULLGRAPH patch installed', flush=True)
'''
n_patch = 0
for c in cells:
    if c['cell_type'] != 'code': continue
    s = ''.join(c['source']); anchor = '_batch_path.write_text(_batch_source)\n'
    if s.count(anchor) == 1:
        c['source'] = [s.replace(anchor, anchor + PATCH)]; n_patch += 1
assert n_patch == 1, n_patch
# 2) P-stage cell -> p_stage2 with candidate graphs
last = cells[-1]; s = ''.join(last['source'])
assert "'p_stage.py'" in s and 'P_CONFIG' in s
s = s.replace("(p / 'p_stage.py').is_file()", "(p / 'p_stage2.py').is_file()")
s = s.replace("str(P_ROOT / 'p_stage.py')", "str(P_ROOT / 'p_stage2.py')")
s = s.replace("'--submission', str(base_sub), '--deadline', str(deadline)]", "'--submission', str(base_sub), '--deadline', str(deadline), '--full', str(work / 'fullgraphs')]")
s = re.sub(r"P_CONFIG = '[^']*'", "P_CONFIG = '%s'" % cfg_file, s)
s = re.sub(r"'variant': '[^']*'", "'variant': '%s'" % variant, s)
s = s.replace("for k in ['movies', 'errors', 'forks_added', 'seconds'] if k in r", "for k in ['movies', 'errors', 'forks_added', 'dsr_added', 'dfork_removed', 'dsr_errors', 'seconds'] if k in r")
assert "'--full'" in s and 'p_stage2.py' in s
last['source'] = [s]
cells[0] = {'cell_type': 'markdown', 'metadata': {}, 'source': [desc]}
for c in cells:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
json.dump(nb, open(out / 'model.ipynb', 'w'), indent=1)
meta = json.load(open(base_meta)); meta.pop('id_no', None)
meta['id'] = 'shawsebastian/' + slug; meta['title'] = title; meta['code_file'] = 'model.ipynb'
meta['is_private'] = True; meta['enable_internet'] = False; meta['enable_gpu'] = True
json.dump(meta, open(out / 'kernel-metadata.json', 'w'), indent=2)
print('built', out, len(cells), 'cells', meta['id'])
