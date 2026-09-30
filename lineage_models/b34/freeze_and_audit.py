"""Freeze calibration-selected candidates before evaluating the reserved audit set."""
from pathlib import Path
import os,json,time,hashlib,subprocess
from concurrent.futures import ThreadPoolExecutor
R=Path('/workspace/biohub');labels={'B3':'localized_motion_cal','B4':'localized_visual_cal'}
for label in labels.values():
    while not (R/'pipeline_evaluation'/label/'report.json').exists():time.sleep(10)
d=json.loads((R/'localized_selection.json').read_text());reports={}
for variant,label in labels.items():
    report=json.loads((R/'pipeline_evaluation'/label/'report.json').read_text())
    assert report['candidate']['n']==18 and report['delta']>.01
    assert report['identity']['selection']==d[variant]
    d[variant]['calibration_frozen']=True
    d[variant]['reference_precision']='full_fp32_all_d4_batched2'
    d[variant]['calibration_report']=label
    reports[variant]={k:report[k] for k in ['candidate','baseline','delta','paired_bootstrap_95']}
dest=R/'frozen_selection.json';assert not dest.exists(),'Never overwrite an audited selection'
dest.write_text(json.dumps(d,indent=2))
manifest={'frozen_unix':time.time(),'selection_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'calibration':reports,'audit_previously_evaluated_for_these_candidates':False,'precision':'Full FP32 detection with all eight D4 transforms; original DeepCenter FP32; lineage same precision as calibration','caveat':'Inherited public detector pretraining overlaps held-out videos. Calibration is exploratory after multiple comparisons. Audit is withheld from new fitting and tuning.'}
(R/'selection_freeze_record.json').write_text(json.dumps(manifest,indent=2));print('SELECTION_FROZEN_BEFORE_AUDIT',json.dumps(manifest),flush=True)
def audit(item):
    gpu,variant=item
    label=variant.lower()+'_frozen_audit';log=R/('evaluate_'+label+'.log')
    subprocess.run(['/workspace/venv/bin/python','-u',str(R/'evaluate_pipeline.py'),'--selection',str(dest),'--variant',variant,'--label',label,'--group','audit'],env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log.open('w'),stderr=subprocess.STDOUT,check=True)
with ThreadPoolExecutor(2) as p:list(p.map(audit,[(0,'B3'),(1,'B4')]))
print('BOTH_FROZEN_AUDITS_COMPLETE',flush=True)
