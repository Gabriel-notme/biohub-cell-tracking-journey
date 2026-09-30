"""Create a B5 notebook variant (cells 1-2 only) with env overrides for post-processing.
usage: make_nb_variant.py <out.ipynb> KEY=VALUE ...   (KEY without BIOHUB_ prefix)
"""
import json, sys, re
out = sys.argv[1]; kv = dict(a.split('=', 1) for a in sys.argv[2:] if '=' in a)
nb = json.load(open('/workspace/nb/B5_local.ipynb'))
c = nb['cells'][2]; s = ''.join(c['source'])
for k, v in kv.items():
    key = 'BIOHUB_' + k
    pat = "os.environ['%s'] = " % key
    n = s.count(pat)
    if n == 0:
        s = "os.environ['%s'] = '%s'\n" % (key, v) + s
    else:
        s = re.sub(re.escape(pat) + r"'[^']*'", pat + "'%s'" % v, s)
    s = re.sub(r"('%s': )[0-9.]+" % re.escape(key), lambda m: m.group(1) + str(float(v)), s)
    print(key, 'occurrences', n)
anchor = "_lineage_stream_proc=subprocess.Popen(_stream_command,env=_stream_environment,stdout=_stream_log,stderr=subprocess.STDOUT)"
assert s.count(anchor) == 1, 'stream anchor'
s = s.replace(anchor, "_lineage_stream_proc=None  # NOSTREAM variant")
c['source'] = [s]
nb['cells'] = nb['cells'][:3]
json.dump(nb, open(out, 'w'))
print('wrote', out)
