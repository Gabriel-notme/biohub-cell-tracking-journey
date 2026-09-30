from pathlib import Path
import os,json,time,argparse
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('POLARS_MAX_THREADS','2');os.environ['BIOHUB_BASE_REPO']='/workspace/biohub/baseline_validation/tracking_repo'
from lineage_pipeline import LineagePipeline
R=Path('/workspace/biohub');TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')
p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args()
base=json.loads((R/'b34_selection.json').read_text())['B4'];selection=json.loads(json.dumps(base));selection['stages']=[s for s in selection['stages'] if s['kind']!='visual_edge'];selection['calibration_frozen']=False
if a.shard==0:(R/'pipeline_bases_b56.json').write_text(json.dumps({'B4':base,'B4_nomotion':selection},indent=2))
model=LineagePipeline(R,selection,TRAIN,R/('baseline_cache_'+str(a.shard)));split=json.loads((R/'track_data/split.json').read_text());dest=R/'reproduced_B4_nomotion';dest.mkdir(exist_ok=True)
for name in (split['calibration']+split['new_audit'])[a.shard::2]:
    output=dest/(name+'.json')
    if output.exists():continue
    source=R/('baseline_graphs' if name in split['calibration'] else 'training_baseline_graphs')/(name+'.json')
    while not source.exists():time.sleep(5)
    raw=json.loads(source.read_text());nn,ee,stats=model.refine(name,{int(k):v for k,v in raw['nodes'].items()},raw['edges'])
    tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps({'nodes':nn,'edges':ee}));os.replace(tmp,output)
    print('REPLACEMENT_INPUT_READY',name,stats['pipeline_seconds'],flush=True)
