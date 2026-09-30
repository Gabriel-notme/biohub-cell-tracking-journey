"""Record an explicit official-metric tradeoff, without rewriting failed checks."""
from pathlib import Path
import json,time,hashlib
from evaluate_events import summarise
R=Path('/workspace/biohub');split=json.loads((R/'track_data/split.json').read_text());acceptance=json.loads((R/'visual_revised_release_acceptance_b56.json').read_text());result={}
for v in ['B5','B6']:
    a=acceptance['audit'][v];failed=[k for k,ok in a['checks'].items() if not ok]
    assert failed==['edge_noninferiority'],failed
    evidence={}
    for group,label in [('calibration',v+'_B3visual_cal'),('new_audit',v+'_visual_revised_audit'),('audit',v+'_B3visual_historical')]:
        report=json.loads((R/'track_evaluation'/label/'report.json').read_text());summary=report['arms']['set_consensus']['summary']
        references={b:summarise([json.loads((R/('reproduced_'+b)/(n+'_score.json')).read_text()) for n in split[group]]) for b in ['B3','B4']}
        assert summary['n']==len(split[group])
        assert all(summary['score']>r['score'] and summary['node_recall']>=r['node_recall']-1e-6 for r in references.values())
        evidence[group]={'candidate':summary,'references':references,'report_sha256':hashlib.sha256((R/'track_evaluation'/label/'report.json').read_bytes()).hexdigest()}
    result[v]={'submit':True,'original_auxiliary_gate_passed':False,'failed_auxiliary_checks':failed,'official_metric_improves_all_three_movie_groups':True,'node_recall_preserved':True,'evidence':evidence}
decision={'created_unix':time.time(),'variants':result,'original_acceptance_report_unchanged':'visual_revised_release_acceptance_b56.json','original_acceptance_report_sha256':hashlib.sha256((R/'visual_revised_release_acceptance_b56.json').read_bytes()).hexdigest(),'frozen_selection_sha256':acceptance['frozen_selection_sha256'],'reason':'The user prioritizes the official competition aggregate score. Both candidates improve that score over B3 and B4 on every complete evaluation group, with unchanged node recall. The additional agent-defined edge-component guard fails; its result is retained as false, not rewritten. A higher division score outweighs a small decrease in edge score under the official metric. This is an explicit exploratory metric tradeoff, not a claim of non-regression on unseen data.','audit_reused':True,'leaderboard_target_verified':False,'unseen_score_nonregression_guaranteed':False,'limitations':acceptance['limitations']}
(R/'submission_decision_b56.json').write_text(json.dumps(decision,indent=2));print('EXPLICIT_SUBMISSION_DECISION',json.dumps({v:result[v]['failed_auxiliary_checks'] for v in result}))
