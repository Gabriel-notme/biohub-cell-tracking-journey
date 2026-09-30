"""Freeze two calibration choices and evaluate their unchanged weights on reserved movies."""
from pathlib import Path
import os,json,time,hashlib,argparse,subprocess
from concurrent.futures import ThreadPoolExecutor
import torch
from evaluate_events import summarise,paired_interval

R=Path('/workspace/biohub');P='/workspace/venv/bin/python'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--choices',required=True);a=p.parse_args()
    choices=json.loads(Path(a.choices).read_text());assert set(choices)=={'B5','B6'}
    destination=R/'revised_selection_b56.json';assert not destination.exists(),'Never overwrite an audited selection'
    split=json.loads((R/'track_data/split.json').read_text());bases=json.loads((R/'pipeline_bases_b56.json').read_text())
    selected={};records={};audit_commands={}
    for variant,choice in choices.items():
        report_path=R/'track_evaluation'/choice['label']/'report.json';report=json.loads(report_path.read_text());identity=report['identity'];arm=choice['arm'];result=report['arms'][arm]
        assert identity['group']=='calibration' and identity['names']==split['calibration']
        assert result['summary']['n']==18 and result['delta']>0,'Calibration does not improve its input baseline'
        calibration_refs={v:summarise([json.loads((R/('reproduced_'+v)/(n+'_score.json')).read_text()) for n in split['calibration']]) for v in ['B3','B4']}
        assert result['summary']['score']>max(q['score'] for q in calibration_refs.values()),'Calibration must beat both original models'
        assert result['summary']['adj_edge_jaccard']>=report['baseline']['adj_edge_jaccard']-.001
        for name,digest in identity['models'].items():
            assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,'Changed selected weights'
            if name.startswith('Track'):
                checkpoint=torch.load(R/name,map_location='cpu',weights_only=False);training=set(checkpoint['protocol']['train'])
                training.update(checkpoint['protocol'].get('augmentation_source_movies',{}).values())
                assert not training.intersection(split['new_audit']+split['calibration']),name
        base=identity['base'];selection=json.loads(json.dumps(bases[base]));kind=identity.get('kind','track_video');model_files=list(identity['models']);config=identity['configs'][arm]
        selection['stages'].append({'kind':kind,'model_files':model_files,'config':config})
        selection.update(calibration_frozen=True,calibration_report=choice['label'],calibration_arm=arm,reference_precision='full_fp32_all_eight_d4_batch4',variant=variant)
        selected[variant]=selection
        frozen={'arm':arm,'config':config,'models_sha256':identity['models'],'input_base':base,'comparison':'B4','created_unix':time.time()}
        frozen_path=R/('revised_'+variant+'_arm.json');frozen_path.write_text(json.dumps(frozen,indent=2))
        records[variant]={'choice':choice,'calibration_summary':result['summary'],'calibration_delta':result['delta'],'calibration_paired_interval':result['paired_interval'],'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),'models_sha256':identity['models']}
        audit_commands[variant]=[P,'-u',str(R/'evaluate_track_stage.py'),'--models',','.join(model_files),'--kind',kind,'--label',variant+'_revised_audit','--base',base,'--comparison','B4','--group','new_audit','--frozen',str(frozen_path)]
    destination.write_text(json.dumps(selected,indent=2))
    freeze={'frozen_unix':time.time(),'selection_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'calibration':records,'new_audit_scores_previously_used_for_candidate_selection':True,'audit_reused':True,'revision_protocol':json.loads((R/'b3_revision_protocol.json').read_text()),'unlabelled_precomputation':{'fixed_encoder_features_only':True,'annotations_or_scores_read':False,'candidate_heads_run':False,'used_for_fitting_or_selection':False},'audit_movies':split['new_audit'],'limitations':'This is a post-audit baseline revision: the initial B4-based candidates failed. Audit results informed the switch to B3, so the repeated 32-movie comparison is exploratory and is NOT independent validation. Original failed audit evidence is retained. Calibration is exploratory after candidate comparisons. New model fitting excludes these 32 audit movies, but inherited B3/B4 and public detector pretraining overlaps them. Fixed encoder features were precomputed without labels, scores or candidate-head inference to shorten the frozen audit. This is not fully out-of-fold validation of the entire pipeline. No leaderboard score is inferred from calibration.'}
    (R/'revised_selection_freeze_b56.json').write_text(json.dumps(freeze,indent=2));print('BOTH_SELECTIONS_FROZEN',json.dumps(records),flush=True)
    def run(item):
        gpu,variant=item
        subprocess.run(audit_commands[variant],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','POLARS_MAX_THREADS':'2'},stdout=open(R/(variant+'_revised_audit.log'),'w'),stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(2) as pool:list(pool.map(run,[(0,'B5'),(1,'B6')]))
    baselines={v:[json.loads((R/('reproduced_'+v)/(n+'_score.json')).read_text()) for n in split['new_audit']] for v in ['B3','B4']}
    baseline_summary={v:summarise(rows) for v,rows in baselines.items()};audits={};passed=True
    for variant,choice in choices.items():
        report=json.loads((R/'track_evaluation'/(variant+'_revised_audit')/'report.json').read_text());result=report['arms'][choice['arm']];rows=result['rows'];summary=result['summary']
        by_embryo={e:{'candidate':summarise([r for r in rows if r['movie'].startswith(e)]),'B4':summarise([r for r in baselines['B4'] if r['movie'].startswith(e)])} for e in ['44b6','6bba']}
        checks={'all_32_movies':summary['n']==32,'beats_B3':summary['score']>baseline_summary['B3']['score'],'beats_B4':summary['score']>baseline_summary['B4']['score'],
                'edge_noninferiority':summary['adj_edge_jaccard']>=baseline_summary['B4']['adj_edge_jaccard']-.001,
                'node_recall_preserved':summary['node_recall']>=baseline_summary['B4']['node_recall']-1e-6,
                'no_material_embryo_regression':all(q['candidate']['score']>=q['B4']['score']-.002 for q in by_embryo.values()),
                'paired_lower_bound':result['paired_interval'][0]>-.005,
                'no_solver_fallback':sum(r.get('refinement',{}).get('solver_fallback_frames',0) for r in rows)==0}
        audits[variant]={'summary':summary,'delta':result['delta'],'paired_interval':result['paired_interval'],'by_embryo':by_embryo,'checks':checks,'passed':all(checks.values())};passed&=all(checks.values())
    acceptance={'passed':bool(passed),'created_unix':time.time(),'audit_reused':True,'initial_audit_passed':False,'initial_failed_audit_file':'initial_release_acceptance_b56.json','audit':audits,'baseline':baseline_summary,'frozen_selection_sha256':freeze['selection_sha256'],'leaderboard_target':.990,'leaderboard_target_verified':False,'actual_score_not_monitored':True,'limitations':freeze['limitations']}
    (R/'revised_release_acceptance_b56.json').write_text(json.dumps(acceptance,indent=2));print('FROZEN_AUDIT_ACCEPTANCE',json.dumps(acceptance),flush=True)
