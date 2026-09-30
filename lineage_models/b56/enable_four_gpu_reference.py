from pathlib import Path
R=Path('/workspace/biohub');p=R/'cloud_baseline.py';s=p.read_text()
a='import os\n';b="import os\nif os.environ.get('BIOHUB_DATA_ROOT','').endswith('/training_videos'):\n    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'\n"
assert 'endswith' not in s[:350]
s=s.replace(a,b,1)
a='worker_count = min(2, available_gpu_count, len(test_stems))'
b="worker_count = min(4 if os.environ.get('BIOHUB_DATA_ROOT','').endswith('/training_videos') else 2, available_gpu_count, len(test_stems))"
assert s.count(a)==1;s=s.replace(a,b);compile(s,str(p),'exec');p.write_text(s)
print('FOUR_GPU_TRAINING_REFERENCE_ENABLED')
