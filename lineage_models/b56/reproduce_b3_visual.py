"""Preserve B3 divisions, then review ordinary links with the fixed visual model."""
from pathlib import Path
import os,json,argparse,time
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('POLARS_MAX_THREADS','2');os.environ['BIOHUB_BASE_REPO']='/workspace/biohub/baseline_validation/tracking_repo'
from visual_edge_refine import VisualEdgeRefiner
from evaluate_events import score_movie,TRAIN
R=Path('/workspace/biohub')
p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args()
split=json.loads((R/'track_data/split.json').read_text());names=split['calibration']+split['new_audit']+split['audit']
source=json.loads((R/'b34_selection.json').read_text())['B4']['stages'][1]
model=VisualEdgeRefiner([R/n for n in source['model_files']],TRAIN,source['config'],R/('b3_visual_cache_'+str(a.shard)))
dest=R/'reproduced_B3visual';dest.mkdir(exist_ok=True)
for name in names[a.shard::4]:
    if (dest/(name+'_score.json')).exists():continue
    raw=json.loads((R/'reproduced_B3'/(name+'.json')).read_text());nodes={int(k):v for k,v in raw['nodes'].items()}
    nn,ee,stats=model.refine(name,nodes,raw['edges'])
    assert nn==nodes,'Visual stage must preserve node positions'
    from collections import Counter
    before=Counter(e['source_id'] for e in raw['edges']);after=Counter(e['source_id'] for e in ee)
    forks={s for s,n in before.items() if n==2}
    assert {(e['source_id'],e['target_id']) for e in raw['edges'] if e['source_id'] in forks}=={(e['source_id'],e['target_id']) for e in ee if e['source_id'] in forks}
    assert {s for s,n in after.items() if n==2}==forks
    score=score_movie(name,nn,ee);score['refinement']=stats
    temporary=dest/(name+'.tmp');temporary.write_text(json.dumps({'nodes':nn,'edges':ee}));os.replace(temporary,dest/(name+'.json'));(dest/(name+'_score.json')).write_text(json.dumps(score,indent=2));print('B3_VISUAL_REPRODUCED',name,stats,flush=True)
