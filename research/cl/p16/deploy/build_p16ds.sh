#!/bin/bash
# Build the P16 stage/dataset dir /workspace/p16ds = deployed p12ds + DivNet second-opinion files + p_stage10.py + p16a/p16b configs.
set -e
D=/workspace/p16ds; S=/workspace/cl/p16
rm -rf $D.tmp; mkdir -p $D.tmp
cp -a /workspace/p12ds/. $D.tmp/; rm -rf $D.tmp/__pycache__
cp $S/divnet/divnet_draft_NOT_ACCEPTED.py $D.tmp/divnet_core.py
cp $S/deploy/divnet16.py $D.tmp/divnet16.py
for i in 0 1 2; do cp $S/divnet/models/dn1_full_trall_s$i.pt $D.tmp/; done
python3 - <<'EOF'
import json
from pathlib import Path
D = Path('/workspace/p16ds.tmp')
s = (D / 'p_stage9.py').read_text()
old = "    cfg = load_cfg(a)\n"
new = old + ("    if cfg.get('divnet'):\n"
             "        import divnet16\n"
             "        dv = cfg['divnet']\n"
             "        divnet16.install(dc, [Path(__file__).parent / f for f in dv['models']], K=int(dv.get('K', 100)), mode=dv.get('mode', 'blend'))\n")
assert s.count(old) == 1
(D / 'p_stage10.py').write_text(s.replace(old, new))
c = json.loads((D / 'p15_config.json').read_text())
for v, mode in (('P16a', 'blend'), ('P16b', 'rerank')):
    c2 = dict(c); c2['variant'] = v
    c2['divnet'] = {'models': ['dn1_full_trall_s%d.pt' % i for i in range(3)], 'K': 100, 'mode': mode}
    (D / ('%s_config.json' % v.lower())).write_text(json.dumps(c2, indent=1))
print('variant key in p15:', c.get('variant'))
EOF
rm -rf $D; mv $D.tmp $D
diff /workspace/p12ds/p_stage9.py $D/p_stage10.py || true
ls -la $D | wc -l; du -sh $D; md5sum $D/p_stage10.py $D/divnet16.py $D/divnet_core.py $D/p16a_config.json $D/p16b_config.json
