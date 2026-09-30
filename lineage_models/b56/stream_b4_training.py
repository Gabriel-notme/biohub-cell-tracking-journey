from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('POLARS_MAX_THREADS','2');os.environ['BIOHUB_BASE_REPO']='/workspace/biohub/baseline_validation/tracking_repo'
import json,time,argparse,fcntl
from lineage_pipeline import LineagePipeline
R=Path('/workspace/biohub');TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--worker',type=int,required=True);args=a.parse_args()
    split=json.loads((R/'track_data/split.json').read_text());names=split['train'];dest=R/'b4_training_graphs';dest.mkdir(exist_ok=True);locks=R/'b4_training_claims';locks.mkdir(exist_ok=True)
    selection=json.loads((R/'b34_selection.json').read_text())['B4'];pipeline=LineagePipeline(R,selection,TRAIN,R/('training_b4_cache_'+str(args.worker)))
    while True:
        pending=[n for n in names if not (dest/(n+'.json')).exists()]
        if not pending:break
        available=[n for n in pending if (R/'training_baseline_graphs'/(n+'.json')).exists()]
        worked=False
        for name in available:
            with (locks/(name+'.lock')).open('a') as f:
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if (dest/(name+'.json')).exists():continue
                raw=json.loads((R/'training_baseline_graphs'/(name+'.json')).read_text())
                nn,ee,stats=pipeline.refine(name,{int(k):v for k,v in raw['nodes'].items()},raw['edges'])
                tmp=dest/(name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps({'nodes':nn,'edges':ee}));os.replace(tmp,dest/(name+'.json'))
                print('B4_TRAIN_GRAPH_READY',name,stats['pipeline_seconds'],flush=True)
                for p in pipeline.cache.rglob('*.npz'):
                    if name in p.name:p.unlink()
                worked=True;break
        if not worked:time.sleep(5)
    print('B4_TRAINING_GRAPHS_COMPLETE',flush=True)
