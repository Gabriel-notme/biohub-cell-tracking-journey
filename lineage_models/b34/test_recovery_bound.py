"""Compare the complete deployment graphs after removing a quadratic ID scan."""
from pathlib import Path
import os,json,time
os.environ['BIOHUB_BASE_REPO']='/workspace/biohub/baseline_validation/tracking_repo'
from lineage_pipeline import LineagePipeline
R=Path('/workspace/biohub');TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')
selection=json.loads((R/'localized_selection.json').read_text());names=json.loads((R/'events/split.json').read_text())['calibration'];records=[]
for variant,label in [('B3','localized_motion_cal'),('B4','localized_visual_cal')]:
    model=LineagePipeline(R,selection[variant],TRAIN,R)
    for name in names:
        expected=R/'pipeline_evaluation'/label/'graphs'/(name+'.json')
        while not expected.exists():time.sleep(5)
        old=json.loads(expected.read_text());d=json.loads((R/'baseline_graphs'/(name+'.json')).read_text());nodes={int(k):v for k,v in d['nodes'].items()}
        nn,ee,stats=model.refine(name,nodes,d['edges'])
        assert nn=={int(k):v for k,v in old['nodes'].items()},(variant,name,'nodes')
        pairs=lambda edges:sorted((int(e['source_id']),int(e['target_id'])) for e in edges)
        assert pairs(ee)==pairs(old['edges']),(variant,name,'edges')
        records.append({'variant':variant,'movie':name,'exact_graph':True,'seconds':stats['pipeline_seconds']});print('RECOVERY_BOUND_EXACT',variant,name,flush=True)
    del model
(R/'recovery_bound_equivalence.json').write_text(json.dumps({'all_exact':True,'records':records},indent=2))
print('RECOVERY_BOUND_EQUIVALENCE_PASSED',len(records),flush=True)
