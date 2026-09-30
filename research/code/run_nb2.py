import os, sys, json, time
from pathlib import Path
nb_path, movies_arg, run_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
movies = [m for m in (open(movies_arg).read().split() if os.path.exists(movies_arg) else movies_arg.split(',')) if m]
data = run_dir / 'data'; data.mkdir(parents=True, exist_ok=True)
for m in movies:
    p = data / (m + '.zarr')
    if not p.exists(): p.symlink_to('/workspace/data/train/' + m + '.zarr')
work = run_dir / 'working'; work.mkdir(parents=True, exist_ok=True)
os.environ['BIOHUB_DATA_ROOT'] = str(data)
os.environ['BIOHUB_WORKDIR'] = str(work)
os.environ.setdefault('BIOHUB_STAGE_DUMP', str(run_dir / 'stages'))
os.chdir(work)
cells = json.load(open(nb_path))['cells']
g = {'__name__': '__main__'}
t0 = time.time()
for i, c in enumerate(cells):
    if c['cell_type'] != 'code': continue
    src = ''.join(c['source']).replace('/kaggle/working', str(work))
    print('=== CELL', i, 'start', round(time.time() - t0, 1), flush=True)
    exec(compile(src, 'cell%d' % i, 'exec'), g)
    print('=== CELL', i, 'done', round(time.time() - t0, 1), flush=True)
print('RUN_NB_COMPLETE', round(time.time() - t0, 1), flush=True)
