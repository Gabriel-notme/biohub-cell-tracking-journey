from pathlib import Path
import os,subprocess,time
R=Path('/workspace/biohub');P='/workspace/venv/bin/python'
def run(args,gpu,log):
    return subprocess.Popen([P,'-u',str(R/'evaluate_track_stage.py'),*args],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'2','POLARS_MAX_THREADS':'2'},stdout=open(R/log,'a'),stderr=subprocess.STDOUT,start_new_session=True)
for stem in ['Track','Track3']:
    jobs=[]
    for gpu,embryo in [('2','44b6'),('3','6bba')]:
        label=stem+'_hold_'+embryo
        while not (R/label/'ready.json').exists():time.sleep(5)
        jobs.append(run(['--models',label+'/best.pt','--label',label,'--hold-embryo',embryo,'--arms','edges_conservative,divisions_strict,joint_strict'],gpu,label+'_eval.log'))
    for j in jobs:assert j.wait()==0,'Cross-embryo full graph evaluation failed'
    while not all((R/(stem+'_full_'+s)/'ready.json').exists() for s in ['137','823']):time.sleep(5)
    models=','.join(stem+'_full_'+s+'/best.pt' for s in ['137','823'])
    job=run(['--models',models,'--label',stem+'_full_cal','--arms','edges_conservative,edges_strong,divisions_strict,divisions_balanced,joint_strict,joint_balanced'],'2',stem+'_full_cal.log')
    assert job.wait()==0,'Full graph calibration failed'
print('INITIAL_TRACK_EVALUATIONS_DONE',flush=True)
