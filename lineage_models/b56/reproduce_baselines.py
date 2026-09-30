from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('POLARS_MAX_THREADS','2');os.environ['BIOHUB_BASE_REPO']='/workspace/biohub/baseline_validation/tracking_repo'
import json,time,argparse,fcntl,hashlib
while not Path('/workspace/biohub/data_ready.json').exists():time.sleep(5)
import torch
from lineage_pipeline import LineagePipeline
from evaluate_events import score_movie,summarise,TRAIN
R=Path('/workspace/biohub')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args()
    while not (R/'data_ready.json').exists() or not (R/'track_data/split.json').exists():time.sleep(5)
    selections=json.loads((R/'b34_selection.json').read_text());split=json.loads((R/'track_data/split.json').read_text())
    names=split['calibration']+split['audit']+split['new_audit'];pending=names[a.shard::2]
    pipelines={v:LineagePipeline(R,selections[v],TRAIN,R/('baseline_cache_'+str(a.shard))) for v in ['B3','B4']}
    for v in pipelines:(R/('reproduced_'+v)).mkdir(exist_ok=True)
    while pending:
        progress=False
        for name in list(pending):
            source=R/('baseline_graphs' if name in split['calibration']+split['audit'] else 'training_baseline_graphs')/(name+'.json')
            if not source.exists():continue
            raw=json.loads(source.read_text())
            for variant,pipeline in pipelines.items():
                dest=R/('reproduced_'+variant);report=dest/(name+'_score.json')
                if report.exists():continue
                nn,ee,stats=pipeline.refine(name,{int(k):v for k,v in raw['nodes'].items()},raw['edges']);row=score_movie(name,nn,ee);row['refinement']=stats
                tmp=dest/(name+'.tmp');tmp.write_text(json.dumps({'nodes':nn,'edges':ee}));os.replace(tmp,dest/(name+'.json'))
                report.write_text(json.dumps(row,indent=2));print('REPRODUCED',variant,name,json.dumps(row),flush=True)
            pending.remove(name);progress=True;break
        if not progress:time.sleep(10)
    print('BASELINES_REPRODUCED_SHARD',a.shard,flush=True)
