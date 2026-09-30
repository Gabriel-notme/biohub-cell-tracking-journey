from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('POLARS_MAX_THREADS','2')
import json,argparse,time,hashlib
from track_video_refine import TrackVideoRefiner
from evaluate_events import score_movie,summarise,paired_interval,TRAIN
R=Path('/workspace/biohub')

CONFIGS={
 'recover_gap':{'heat_threshold':.4,'edge_threshold':.999,'fork_threshold':.995,'allow_division':False,'bridge_gap':1,'arrival_threshold':.95,'proposal_min_probability':.1,'max_distance':16.},
 'recover_lineage':{'heat_threshold':.4,'edge_threshold':.999,'fork_threshold':.995,'allow_division':True,'bridge_gap':1,'arrival_threshold':.95,'proposal_min_probability':.1,'max_distance':16.},
 'recover_gap2':{'heat_threshold':.4,'edge_threshold':.999,'fork_threshold':.995,'allow_division':False,'bridge_gap':2,'arrival_threshold':.95,'proposal_min_probability':.1,'max_distance':16.},
 'wide_preserve':{'edge_weight':4.,'edge_change_penalty':.8,'fork_threshold':1.1,'fork_veto':0.,'protected_division_radius':2,'endpoint_recovery':True,'endpoint_probability':.999,'disappearance_penalty':50.,'endpoint_min_history':1,'endpoint_min_future':2,'endpoint_motion_error':7.,'endpoint_max_distance':12.,'max_distance':18.,'max_candidates':8,'fork_max_distance':16.},
 'wide_joint':{'edge_weight':4.,'edge_change_penalty':.8,'fork_threshold':.995,'fork_veto':0.,'fork_change_penalty':1.,'new_fork_min_edge':.75,'endpoint_recovery':True,'endpoint_probability':.999,'disappearance_penalty':50.,'endpoint_min_history':1,'endpoint_min_future':2,'endpoint_motion_error':7.,'endpoint_max_distance':12.,'max_distance':18.,'max_candidates':8,'fork_max_distance':16.},
 'learned_continuity':{'edge_weight':4.,'edge_change_penalty':.8,'fork_threshold':1.1,'fork_veto':0.,'protected_division_radius':2,'endpoint_recovery':True,'endpoint_probability':.999,'disappearance_penalty':50.,'endpoint_min_history':1,'endpoint_min_future':2,'endpoint_motion_error':7.,'endpoint_max_distance':12.},
 'fusion_div95':{'mode':'fork_only','edge_weight':0.,'fork_threshold':.95,'fork_veto':0.,'fork_change_penalty':.5},
 'fusion_joint95':{'edge_weight':4.,'edge_change_penalty':.8,'fork_threshold':.95,'fork_veto':0.,'fork_change_penalty':1.,'new_fork_min_edge':.75,'endpoint_recovery':True,'endpoint_probability':.995,'disappearance_penalty':50.},
 'edges_preserve':{'edge_weight':4.,'edge_change_penalty':.8,'fork_threshold':1.1,'fork_veto':0.,'protected_division_radius':2,'endpoint_recovery':True,'endpoint_probability':.995,'disappearance_penalty':50.},
 'joint_preserve':{'edge_weight':4.,'edge_change_penalty':.8,'fork_threshold':.995,'fork_veto':0.,'fork_change_penalty':1.,'new_fork_min_edge':.75,'endpoint_recovery':True,'endpoint_probability':.995,'disappearance_penalty':50.},
 'continuity':{'edge_weight':4.,'edge_change_penalty':.8,'fork_threshold':1.1,'fork_veto':0.,'protected_division_radius':2,'endpoint_recovery':True,'endpoint_probability':.995},
 'joint_wide':{'edge_weight':4.,'edge_change_penalty':.8,'fork_threshold':.995,'fork_veto':0.,'fork_change_penalty':1.,'max_distance':16.,'fork_max_distance':16.,'max_candidates':5,'endpoint_recovery':True,'endpoint_probability':.995},
 'edges_conservative':{'edge_weight':4.,'edge_change_penalty':1.0,'fork_threshold':1.1,'fork_veto':0.,'protected_division_radius':2},
 'edges_strong':{'edge_weight':8.,'edge_change_penalty':.4,'fork_threshold':1.1,'fork_veto':0.,'protected_division_radius':2},
 'divisions_strict':{'mode':'fork_only','edge_weight':0.,'fork_threshold':.995,'fork_veto':0.,'fork_change_penalty':.5},
 'divisions_balanced':{'mode':'fork_only','edge_weight':0.,'fork_threshold':.98,'fork_veto':0.,'fork_change_penalty':.5},
 'joint_strict':{'edge_weight':4.,'edge_change_penalty':.8,'fork_threshold':.995,'fork_veto':0.,'fork_change_penalty':1.},
 'joint_balanced':{'edge_weight':8.,'edge_change_penalty':.5,'fork_threshold':.98,'fork_veto':0.,'fork_change_penalty':.5},
}
CONFIGS['wide_fusion95']={**CONFIGS['wide_joint'],'fork_threshold':.95}
CONFIGS['recover_image_strict']={**CONFIGS['recover_gap'],'heat_threshold':.6,'arrival_threshold':.99}
CONFIGS['recover_image_balanced']={**CONFIGS['recover_gap'],'heat_threshold':.6,'edge_threshold':.995,'arrival_threshold':.9}
CONFIGS['recover_image_lineage']={**CONFIGS['recover_image_balanced'],'allow_division':True,'fork_threshold':.995}
CONFIGS['consensus_edges']={**CONFIGS['edges_preserve'],'edge_consensus_gate':True,'edge_old_max':.5,'edge_new_min':.995}
CONFIGS['consensus_wide']={**CONFIGS['wide_preserve'],'edge_consensus_gate':True,'edge_old_max':.5,'edge_new_min':.995}
CONFIGS['consensus_joint']={**CONFIGS['wide_joint'],'edge_consensus_gate':True,'edge_old_max':.5,'edge_new_min':.995}
CONFIGS['consensus_strict']={**CONFIGS['wide_preserve'],'edge_consensus_gate':True,'edge_old_max':.2,'edge_new_min':.999}
for key,threshold in [('98',.98),('90',.90)]:
    CONFIGS['endpoint_only'+key]={**CONFIGS['wide_preserve'],'edge_consensus_gate':True,'edge_old_max':-1.,'edge_new_min':1.1,'endpoint_probability':threshold,'endpoint_min_history':3,'endpoint_min_future':3,'endpoint_motion_error':4.5,'endpoint_max_distance':8.,'edge_change_penalty':.1}
CONFIGS['review_divisions']={'mode':'fork_only','edge_weight':4.,'fork_threshold':1.1,'fork_veto':.05,'fork_change_penalty':1.,'disappearance_penalty':50.}
CONFIGS['review_divisions_strict']={**CONFIGS['review_divisions'],'fork_veto':.01}
CONFIGS['repair_divisions']={'mode':'fork_only','edge_weight':4.,'fork_threshold':1.1,'fork_veto':0.,'fork_repair':True,'fork_repair_threshold':.2,'fork_change_penalty':1.,'disappearance_penalty':50.,'max_candidates':8,'max_distance':18.,'fork_max_distance':16.}
CONFIGS['specialist_edges']={**CONFIGS['wide_preserve'],'edge_expert':0,'fork_expert':1}
CONFIGS['specialist_joint']={**CONFIGS['wide_joint'],'edge_expert':0,'fork_expert':1}
CONFIGS['specialist_review']={**CONFIGS['review_divisions'],'edge_expert':0,'fork_expert':1}
CONFIGS['flow_strict']={**CONFIGS['consensus_wide'],'endpoint_reassignment':True,'endpoint_old_max':.2,'endpoint_probability':.999}
CONFIGS['flow_balanced']={**CONFIGS['consensus_wide'],'endpoint_reassignment':True,'endpoint_old_max':.5,'endpoint_probability':.995}
for threshold in [.8,.5,.2]:
    CONFIGS['fusion_fork'+str(int(threshold*100))]={**CONFIGS['fusion_div95'],'fork_threshold':threshold,'max_candidates':8,'max_distance':18.,'fork_max_distance':16.,'new_fork_min_edge':.75}
CONFIGS['division_reassign_strict']={**CONFIGS['wide_joint'],'mode':'fork_only','division_reassignment':True,'division_old_max':.2}
CONFIGS['division_reassign_fusion']={**CONFIGS['fusion_fork80'],'edge_weight':4.,'division_reassignment':True,'division_old_max':.2,'disappearance_penalty':50.}
CONFIGS['prune_rejected01']={**CONFIGS['consensus_wide'],'edge_low_confidence_pruning':True,'edge_pruning_max':.01}
CONFIGS['prune_rejected05']={**CONFIGS['consensus_wide'],'edge_low_confidence_pruning':True,'edge_pruning_max':.05}
for key,threshold in [('995',.995),('98',.98),('95',.95),('80',.8)]:
    CONFIGS['division_direct'+key]={**CONFIGS['fusion_div95'],'fork_threshold':threshold,'new_fork_min_edge':0.,'max_candidates':8,'max_distance':18.,'fork_max_distance':16.}
for key in ['995','98','95','80']:CONFIGS['division_birth'+key]={**CONFIGS['division_direct'+key],'fork_from_endpoints':True}
for key,threshold in [('995',.995),('95',.95)]:
    CONFIGS['division_local'+key]={**CONFIGS['joint_strict'],'division_local_assignment':True,'fork_threshold':threshold,'max_candidates':8,'max_distance':18.,'fork_max_distance':16.,'new_fork_min_edge':.5}
for blend in [.25,.5,.75]:
    CONFIGS['centroid_residual'+str(int(blend*100))]={'centroid_blend':blend,'centroid_max_shift':2.,'centroid_max_error':2.,'centroid_min_separation':2.}
CONFIGS['set_consensus']={**CONFIGS['consensus_wide'],'edge_new_min':.8,'edge_old_max':.2}
for key,threshold in [('95',.95),('80',.8),('50',.5)]:
    CONFIGS['set_joint'+key]={**CONFIGS['wide_joint'],'fork_threshold':threshold,'new_fork_min_edge':.5}
CONFIGS['set_fork20']={**CONFIGS['division_direct80'],'fork_threshold':.2,'new_fork_min_edge':0.}
CONFIGS['set_local20']={**CONFIGS['division_local95'],'fork_threshold':.2,'new_fork_min_edge':.1}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['track_ensemble','track_set_sparse','track_rank','track_set','track_video','track_fusion','track_recovery','centroid'],default='track_video');p.add_argument('--models',required=True);p.add_argument('--label',required=True);p.add_argument('--base',default='B4');p.add_argument('--comparison');p.add_argument('--arms',default='edges_conservative,divisions_strict,joint_strict');p.add_argument('--group',default='calibration');p.add_argument('--frozen');p.add_argument('--hold-embryo',default='');a=p.parse_args()
    split=json.loads((R/'track_data/split.json').read_text());names=split[a.group]
    if a.hold_embryo:names=[n for n in names if n.startswith(a.hold_embryo)]
    configs={k:CONFIGS[k] for k in a.arms.split(',')}
    if a.frozen:
        frozen=json.loads(Path(a.frozen).read_text());configs={frozen['arm']:frozen['config']}
    if a.group=='new_audit':assert a.frozen,'New audit requires frozen configuration'
    out=R/'track_evaluation'/a.label;out.mkdir(parents=True,exist_ok=True)
    paths=[R/x for x in a.models.split(',')]
    from lineage_pipeline import refiner_class
    refiner=refiner_class(a.kind)(paths,TRAIN,cache_dir=R/'track_prediction_cache')
    identity={'kind':a.kind,'models':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'base':a.base,'group':a.group,'names':names,'configs':configs}
    if a.frozen:
        assert frozen['models_sha256']==identity['models'] and frozen['input_base']==a.base and frozen['comparison']==(a.comparison or a.base),'Frozen experiment identity mismatch'
    if a.comparison:identity['comparison']=a.comparison
    if a.kind=='track_video':identity.pop('kind')
    if (out/'identity.json').exists():assert json.loads((out/'identity.json').read_text())==identity,'Changed experiment identity'
    (out/'identity.json').write_text(json.dumps(identity,indent=2));base=[];rows={k:[] for k in configs}
    for name in names:
        source=R/('reproduced_'+a.base)/(name+'.json');bscore=R/('reproduced_'+(a.comparison or a.base))/(name+'_score.json')
        while not bscore.exists() or not source.exists() or not (R/('reproduced_'+a.base)/(name+'_score.json')).exists():time.sleep(5)
        base.append(json.loads(bscore.read_text()));raw=json.loads(source.read_text())
        for arm,config in configs.items():
            dest=out/arm;dest.mkdir(exist_ok=True);report=dest/(name+'.json')
            if report.exists():row=json.loads(report.read_text())
            else:
                from joint_refine import JOINT_DEFAULT
                from recover_joint import RECOVERY_DEFAULT
                refiner.config={**(RECOVERY_DEFAULT if a.kind=='track_recovery' else JOINT_DEFAULT),**config};nn,ee,stats=refiner.refine(name,{int(k):v for k,v in raw['nodes'].items()},raw['edges'])
                row=score_movie(name,nn,ee);row['refinement']=stats;report.write_text(json.dumps(row,indent=2))
                gd=dest/'graphs';gd.mkdir(exist_ok=True);(gd/(name+'.json')).write_text(json.dumps({'nodes':nn,'edges':ee}))
                print('TRACK_ARM_SCORED',arm,name,json.dumps(row),flush=True)
            rows[arm].append(row)
        (out/'progress.json').write_text(json.dumps({'n':len(base),'total':len(names),'baseline':summarise(base),'arms':{k:summarise(v) for k,v in rows.items()}},indent=2))
    report={'identity':identity,'baseline':summarise(base),'arms':{}}
    for arm,rr in rows.items():
        result=summarise(rr);report['arms'][arm]={'summary':result,'delta':result['score']-report['baseline']['score'],'paired_interval':paired_interval(base,rr),'by_embryo':{e:summarise([r for r in rr if r['movie'].startswith(e)]) for e in ['44b6','6bba'] if any(r['movie'].startswith(e) for r in rr)},'rows':rr}
    (out/'report.json').write_text(json.dumps(report,indent=2));print('TRACK_EVALUATION_DONE',json.dumps({k:{'summary':v['summary'],'delta':v['delta']} for k,v in report['arms'].items()}),flush=True)
