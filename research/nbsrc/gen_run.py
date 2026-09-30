"""gen_run.py <tag> <cuda devices> <data dir> [KEY=VALUE ...]: build /workspace/nbrun/<tag>/run.py from the P20 notebook cells 1-3
(base tracker + B5 lineage), with /kaggle/working -> /workspace/nbrun/<tag>/working and the notebook's hard-coded
os.environ['KEY'] = '...' lines replaced by the given values (each must occur exactly once)."""
import sys
from pathlib import Path
tag, gpus, data = sys.argv[1:4]; ov = dict(a.split('=', 1) for a in sys.argv[4:])
src = '\n'.join(Path('/workspace/nbsrc/cell%d.py' % i).read_text() for i in (1, 2, 3))
work = '/workspace/nbrun/%s/working' % tag
for k, v in ov.items():
    import re
    pat = re.compile(r"os\.environ\['%s'\] = '[^']*'" % re.escape(k))
    n = len(pat.findall(src)); assert n == 1, (k, n)
    src = pat.sub("os.environ['%s'] = '%s'" % (k, v), src)
src = src.replace('/kaggle/working', work)
head = "import os\nos.environ['BIOHUB_DATA_ROOT'] = %r\nos.environ['CUDA_VISIBLE_DEVICES'] = %r\n" % (data, gpus)
out = Path('/workspace/nbrun/%s' % tag); (out / 'working').mkdir(parents=True, exist_ok=True)
(out / 'run.py').write_text(head + src)
print('wrote', out / 'run.py', 'overrides', ov)
