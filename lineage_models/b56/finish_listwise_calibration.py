from pathlib import Path
import os,time,subprocess
R=Path('/workspace/biohub');P='/workspace/venv/bin/python';deadline=time.time()+1200
while not (R/'TrackRankListwiseRobust/ready.json').exists():
    assert time.time()<deadline,'Training readiness timeout'
    time.sleep(10)
subprocess.run([P,'-u',str(R/'evaluate_track_stage.py'),'--kind','track_rank','--models','TrackRankListwiseRobust/best.pt,TrackRobust_137/best.pt','--label','B5_Listwise_B3_cal','--base','B3','--comparison','B3','--group','calibration','--arms','set_consensus'],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':'2'},stdout=open(R/'B5_Listwise_B3_cal.log','w'),stderr=subprocess.STDOUT,check=True)
