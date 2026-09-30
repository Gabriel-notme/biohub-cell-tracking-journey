"""Add 'frag_reinsert' to p_stage4.py: re-insert dropped short chains before the link steps, remove unattached ones after."""
from pathlib import Path
import json, shutil
p = Path('/workspace/p56stage/p_stage4.py'); s = p.read_text()
a1 = "            cdir = Path(a.config).parent\n"
assert s.count(a1) == 1
add1 = """            ri_new = None; ri_backup = (nodes2, ne)
            if cfg.get('frag_reinsert') is not None and fullp is not None and fullp.exists():
                try:
                    import frag_reinsert, edge_link
                    fr = cfg['frag_reinsert']
                    n9, e9, ri_new, st9 = frag_reinsert.add(nodes2, ne, edge_link.load_full(fullp), max_len=int(fr.get('max_len', 5)), clear_um=float(fr.get('clear_um', 4.0)))
                    nodes2, ne = n9, e9; st = dict(st, **st9)
                except Exception as e:
                    ri_new = None; nodes2, ne = ri_backup; st = dict(st, ri_error=repr(e)[:300])
""" + a1
s = s.replace(a1, add1)
a2 = "            if cfg.get('post_prune'):\n"
assert s.count(a2) == 1
add2 = """            if ri_new is not None:
                try:
                    import frag_reinsert
                    n10, e10, st10 = frag_reinsert.cleanup(nodes2, ne, ri_new)
                    check_graph(n10, e10, nodes)
                    nodes2, ne = n10, e10; st = dict(st, **st10)
                except Exception as e:
                    nodes2, ne = ri_backup; st = dict(st, ri_cleanup_error=repr(e)[:300])
""" + a2
s = s.replace(a2, add2)
p.write_text(s)
shutil.copy('/workspace/cl/frag_reinsert.py', '/workspace/p56stage/frag_reinsert.py')
e = json.load(open('/workspace/p56stage/p7e_config.json'))
f = dict(e, variant='P7f', frag_reinsert={'max_len': 5, 'clear_um': 4.0})
json.dump(f, open('/workspace/p56stage/p7f_config.json', 'w'), indent=1)
print(json.dumps(f))
