from pathlib import Path
import json,argparse
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--variant',required=True,choices=['B5','B6']);a=p.parse_args()
cohort=json.loads((R.parent/'round4/runtime_cohort.json').read_text());names=[r['movie'] for r in cohort['records']]
source=R/a.variant.lower();out=R/(a.variant.lower()+'_final_cohort');out.mkdir(exist_ok=True)
nb=json.loads((source/'model.ipynb').read_text())
setup="""from pathlib import Path
import os,json
cohort_names=%r
train=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')
if not train.exists():train=Path('/kaggle/input/biohub-cell-tracking-during-development/train')
assert train.exists()
dest=Path('/kaggle/working/runtime_cohort_videos');dest.mkdir(exist_ok=True)
for name in cohort_names:
    path=dest/(name+'.zarr')
    if not path.exists():path.symlink_to(train/(name+'.zarr'),target_is_directory=True)
os.environ['BIOHUB_DATA_ROOT']=str(dest)
print('Fixed runtime-only cohort; no labels are read:',cohort_names)
"""%names
nb['cells'].insert(1,{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':setup.splitlines(True)})
nb['cells'][0]['source']=['# Frozen '+a.variant+' sixteen-movie runtime benchmark\nAll model and source hashes match the final candidate. This notebook measures runtime and output structure only; no ground-truth labels are read. It is not a competition submission.']
for i,c in enumerate(nb['cells']):c['id']='frozen-cohort-'+str(i)
(out/'model.ipynb').write_text(json.dumps(nb),encoding='utf-8')
meta=json.loads((source/'kernel-metadata.json').read_text());meta.update(id='shawsebastian/biohub-'+a.variant.lower()+'-frozen-runtime',title='Biohub '+a.variant+' Frozen Runtime');meta.pop('id_no',None)
(out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2));print(meta['id'])
