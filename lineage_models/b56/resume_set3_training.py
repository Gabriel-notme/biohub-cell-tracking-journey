from pathlib import Path
import os,subprocess
R=Path('/workspace/biohub');P='/workspace/venv/bin/python';jobs=[]
for shard,gpu in [(0,'0'),(1,'1')]:jobs.append(subprocess.Popen([P,'-u',str(R/'prepare_set_features.py'),'--shard',str(shard)],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu},stdout=open(R/('set_prepare_'+str(shard)+'.log'),'a'),stderr=subprocess.STDOUT,start_new_session=True))
for job in jobs:assert job.wait()==0
jobs=[]
for seed,gpu in [(137,'0'),(823,'1')]:jobs.append(subprocess.Popen([P,'-u',str(R/'train_track_set.py'),'--variant','Set3','--seed',str(seed)],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu},stdout=open(R/('TrackSet3_'+str(seed)+'.log'),'a'),stderr=subprocess.STDOUT,start_new_session=True))
for job in jobs:assert job.wait()==0
print('SET3_TRAINING_DONE',flush=True)
