#!/bin/bash
# Build /workspace/p17ds = /workspace/p16ds + p17_post.py + p_stage11.py (p_stage10 + P17 clean-up block at the end of each movie)
# + p17_config.json (P15 + clean-up) and p18_config.json (P16a DivNet blend + clean-up).
set -e
D=/workspace/p17ds
rm -rf $D.tmp; mkdir -p $D.tmp
cp -a /workspace/p16ds/. $D.tmp/; rm -rf $D.tmp/__pycache__
cp /workspace/cl/p16/deploy/p17_post.py $D.tmp/
python3 - <<'EOF'
import json
from pathlib import Path
D = Path('/workspace/p17ds.tmp')
s = (D / 'p_stage10.py').read_text()
old = "            pairs = [(int(e['source_id']), int(e['target_id'])) for e in ne]\n"
new = ("            if cfg.get('p17') is not None:\n"
       "                try:\n"
       "                    import p17_post\n"
       "                    refp17 = Path(a.graphs).parent / 'reference_graphs' / (name + '.json')\n"
       "                    n17, e17, st17 = p17_post.apply(nodes2, ne, refp17, **cfg['p17'])\n"
       "                    check_graph(n17, e17, nodes)\n"
       "                    nodes2, ne = n17, e17; st = dict(st, **st17)\n"
       "                except Exception as e:\n"
       "                    st = dict(st, p17_error=repr(e)[:300])\n") + old
assert s.count(old) == 1
(D / 'p_stage11.py').write_text(s.replace(old, new))
p17 = {'cutdup_on': 1, 'forkfrag_on': 1, 'shortbranch_on': 1}
c = json.loads((D / 'p15_config.json').read_text()); c['variant'] = 'P17'; c['p17'] = p17
(D / 'p17_config.json').write_text(json.dumps(c, indent=1))
c = json.loads((D / 'p16a_config.json').read_text()); c['variant'] = 'P18'; c['p17'] = p17
(D / 'p18_config.json').write_text(json.dumps(c, indent=1))
EOF
rm -rf $D; mv $D.tmp $D
diff /workspace/p16ds/p_stage10.py $D/p_stage11.py || true
du -sh $D; md5sum $D/p_stage11.py $D/p17_post.py $D/p17_config.json $D/p18_config.json
