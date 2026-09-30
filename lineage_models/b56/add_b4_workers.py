from pathlib import Path
import os,subprocess
R=Path('/workspace/biohub')
for worker in range(4,8):
    subprocess.Popen(['/workspace/venv/bin/python','-u',str(R/'stream_b4_training.py'),'--worker',str(worker)],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(worker%4)},stdout=open(R/f'stream_b4_train_{worker}.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
print('FOUR_ADDITIONAL_B4_WORKERS_STARTED')
