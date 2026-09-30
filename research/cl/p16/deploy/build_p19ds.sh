#!/bin/bash
# Build /workspace/p19ds = /workspace/p17ds + p19_dup.py + p19_edge_deploy.py + p_stage12.py (p_stage11 + P19 small clean-up block
# after the P17 block) + configs: p19s (P17 + small), p19a (P17 + small + DivNet blend), p19b (P17 + small + DivNet rerank).
set -e
D=/workspace/p19ds
rm -rf $D.tmp; mkdir -p $D.tmp
cp -a /workspace/p17ds/. $D.tmp/; rm -rf $D.tmp/__pycache__
cp /workspace/cl/ideas/p19_dup.py /workspace/cl/ideas/p19_edge_deploy.py $D.tmp/
python3 - <<'EOF'
import json
from pathlib import Path
D = Path('/workspace/p19ds.tmp')
s = (D / 'p_stage11.py').read_text()
old = "            pairs = [(int(e['source_id']), int(e['target_id'])) for e in ne]\n"
new = ("            if cfg.get('p19small') is not None:\n"
       "                try:\n"
       "                    import p19_dup, p19_edge_deploy, zarr as _zarr19\n"
       "                    ps = cfg['p19small']\n"
       "                    refp19 = Path(a.graphs).parent / 'reference_graphs' / (name + '.json')\n"
       "                    n19, e19 = nodes2, ne\n"
       "                    if ps.get('dup', 1):\n"
       "                        n19, e19, _s19 = p19_dup.post(n19, e19, refp19, start_r=2.5, par_r=3.5, tt_join=None)\n"
       "                    if ps.get('border', 1):\n"
       "                        shp19 = tuple(int(x) for x in _zarr19.open_group(str(Path(a.data) / (name + '.zarr')), mode='r')['0'].shape[-2:])\n"
       "                        r19 = p19_edge_deploy.yx_border_stubs(n19, e19, shape_yx=shp19); n19, e19 = r19[0], r19[1]\n"
       "                    check_graph(n19, e19, nodes)\n"
       "                    st = dict(st, p19_removed=len(nodes2) - len(n19)); nodes2, ne = n19, e19\n"
       "                except Exception as e:\n"
       "                    st = dict(st, p19_error=repr(e)[:300])\n") + old
assert s.count(old) == 1 and s.index("if cfg.get('p17')") < s.index(old)
(D / 'p_stage12.py').write_text(s.replace(old, new))
small = {'dup': 1, 'border': 1}
c = json.loads((D / 'p17_config.json').read_text()); c['variant'] = 'P19S'; c['p19small'] = small
(D / 'p19s_config.json').write_text(json.dumps(c, indent=1))
c = json.loads((D / 'p18_config.json').read_text()); c['variant'] = 'P19A'; c['p19small'] = small
(D / 'p19a_config.json').write_text(json.dumps(c, indent=1))
c = json.loads((D / 'p18_config.json').read_text()); c['variant'] = 'P19B'; c['p19small'] = small; c['divnet'] = dict(c['divnet'], mode='rerank')
(D / 'p19b_config.json').write_text(json.dumps(c, indent=1))
EOF
rm -rf $D; mv $D.tmp $D
diff /workspace/p17ds/p_stage11.py $D/p_stage12.py | head -30 || true
du -sh $D; md5sum $D/p_stage12.py $D/p19_dup.py $D/p19_edge_deploy.py $D/p19*_config.json
python3 -c "import json; [print(f, {k: json.load(open('/workspace/p19ds/'+f)).get(k) for k in ('variant','divnet','p17','p19small')}) for f in ('p19s_config.json','p19a_config.json','p19b_config.json')]"
