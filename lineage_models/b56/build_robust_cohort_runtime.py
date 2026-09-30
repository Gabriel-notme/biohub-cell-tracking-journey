from pathlib import Path
import json
R=Path('work/round5');cohort=json.loads(Path('work/round4/runtime_cohort.json').read_text());names=[r['movie'] for r in cohort['records']]
out=R/'robust_cohort_runtime';out.mkdir(exist_ok=True);nb=json.loads((R/'set_runtime/model.ipynb').read_text())
setup="""from pathlib import Path
import os,json
cohort_names=%r
train=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')
if not train.exists():train=Path('/kaggle/input/biohub-cell-tracking-during-development/train')
assert train.exists()
dest=Path('/kaggle/working/runtime_cohort_videos');dest.mkdir(exist_ok=True)
for name in cohort_names:
    p=dest/(name+'.zarr')
    if not p.exists():p.symlink_to(train/(name+'.zarr'),target_is_directory=True)
os.environ['BIOHUB_DATA_ROOT']=str(dest)
print('Runtime-only fixed size cohort:',cohort_names)
"""%names
nb['cells'].insert(1,{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':setup.splitlines(True)})
nb['cells'][0]['source']=['# B5/B6 sixteen-movie runtime benchmark\nFixed training-video size quantiles measure uncached inference cost only. No performance labels are read. Not a competition submission.']
for i,c in enumerate(nb['cells']):c['id']='cohort-'+str(i)
(out/'model.ipynb').write_text(json.dumps(nb),encoding='utf-8')
meta=json.loads((R/'set_runtime/kernel-metadata.json').read_text());meta.update(id='shawsebastian/biohub-b56-robust-sixteen-movie-runtime',title='Biohub B56 Robust Sixteen Movie Runtime');meta.pop('id_no',None)
(out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
