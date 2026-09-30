from pathlib import Path
import json,time,os
from teacher_branch import merge_teachers
from evaluate_events import score_movie,summarise
R=Path('/workspace/biohub');refs=json.loads((R/'b34_selection.json').read_text());b3=refs['B3'];b4=refs['B4']
branch={'kind':'teacher_branch','model_files':[b3['stages'][1]['model_files'][0],*b4['stages'][1]['model_files'],*b3['stages'][2]['model_files'],*b3['stages'][3]['model_files']],
        'config':{'motion':b3['stages'][1]['config'],'visual':b4['stages'][1]['config'],'recovery':b3['stages'][2]['config'],'centroid':b3['stages'][3]['config']}}
base={'base':b3['base'],'stages':[b3['stages'][0],branch],'calibration_frozen':False}
bases=json.loads((R/'pipeline_bases_b56.json').read_text());bases['Teachers']=base;(R/'pipeline_bases_b56.json').write_text(json.dumps(bases,indent=2))
protocol={'created_unix':time.time(),'proposal':'Shared B1/B2 event and division-review trunk, then B3 motion/recovery and B4 visual/recovery branches. Retain all B4 ordinary links except incoming conflicts required by a compatible B3 division whose existing B4 child already agrees. Never change B4 divisions. Require exact aligned node dictionaries; otherwise return B4.','thresholds_fit':False,'audit_reused':True,'numeric_guards_relaxed':False}
(R/'teacher_revision_protocol.json').write_text(json.dumps(protocol,indent=2));split=json.loads((R/'track_data/split.json').read_text());dest=R/'reproduced_Teachers';dest.mkdir(exist_ok=True)
for group in ['calibration','new_audit','audit']:
    rows=[]
    for name in split[group]:
        if (dest/(name+'_score.json')).exists():rows.append(json.loads((dest/(name+'_score.json')).read_text()));continue
        a=json.loads((R/'reproduced_B3'/(name+'.json')).read_text());b=json.loads((R/'reproduced_B4'/(name+'.json')).read_text())
        nn,ee,stats=merge_teachers({int(k):v for k,v in a['nodes'].items()},a['edges'],{int(k):v for k,v in b['nodes'].items()},b['edges'])
        temporary=dest/(name+'.tmp');temporary.write_text(json.dumps({'nodes':nn,'edges':ee}));os.replace(temporary,dest/(name+'.json'))
        row=score_movie(name,nn,ee);row['refinement']=stats;rows.append(row);(dest/(name+'_score.json')).write_text(json.dumps(row,indent=2))
    print('TEACHER_BASE_COMPLETE',group,json.dumps(summarise(rows)),flush=True)
