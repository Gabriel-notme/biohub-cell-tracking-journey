from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,os,subprocess
R=Path('/workspace/biohub');P='/workspace/venv/bin/python'
labels=['Track3_fusion_cal_division','Track3_fusion_cal_expanded','Track3_fusion_cal_consensus','Track3_fusion_cal_directdivision','Track3_fusion_local_division','Track3_fusion_endpoint_completion']
def run(label):
    d=R/'track_evaluation'/label
    if (d/'report.json').exists():return
    if (d/'identity.json').exists():identity=json.loads((d/'identity.json').read_text());arms=list(identity['configs']);models=list(identity['models'])
    else:
        arms=['division_local995','division_local95'] if label.endswith('local_division') else ['endpoint_only98','endpoint_only90'];models=['Track3_fusion/best.pt','Track3_b4_137/best.pt']
    command=[P,'-u',str(R/'evaluate_track_stage.py'),'--kind','track_fusion','--models',','.join(models),'--label',label,'--arms',','.join(arms)]
    if (d/'identity.json').exists():
        command+=['--base',identity['base']]
        if 'comparison' in identity:command+=['--comparison',identity['comparison']]
    subprocess.run(command,cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','OMP_NUM_THREADS':'2','POLARS_MAX_THREADS':'2'},stdout=open(R/(label+'.log'),'a'),stderr=subprocess.STDOUT,check=True)
with ThreadPoolExecutor(2) as pool:list(pool.map(run,labels))
