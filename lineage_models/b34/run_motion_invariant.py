from pathlib import Path
import subprocess,os,json,time
R=Path('/workspace/biohub');PY='/workspace/venv/bin/python'
cmd=[PY,'-u',str(R/'train_motion.py'),'--out',str(R/'Motion_invariant'),'--rotation-invariant']
subprocess.run(cmd,stdout=(R/'train_motion_invariant.log').open('w'),stderr=subprocess.STDOUT,check=True)
cmd=[PY,'-u',str(R/'evaluate_joint.py'),'--models',str(R/'Motion_invariant/best.pt'),'--refiner','motion','--label','motion_invariant_cal','--configs',str(R/'motion_configs.json')]
subprocess.run(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},stdout=(R/'evaluate_motion_invariant.log').open('w'),stderr=subprocess.STDOUT,check=True)
print('INVARIANT_MOTION_CALIBRATION_COMPLETE',flush=True)
