"""Add 'post_prune' to p_stage4.py: when set (and prune_min_len is 0/absent), isolated-node pruning runs after the link steps."""
from pathlib import Path
import json
p = Path('/workspace/p56stage/p_stage4.py'); s = p.read_text()
anchor = "            if cfg.get('dc_pass2'):\n"
assert s.count(anchor) == 1
add = """            if cfg.get('post_prune'):
                try:
                    import prune
                    n8, e8, st8 = prune.prune_fragments(nodes2, ne, int(cfg['post_prune']))
                    check_graph(n8, e8, nodes)
                    nodes2, ne = n8, e8; st = dict(st, post_pruned=st8.get('pruned_nodes', 0))
                except Exception as e:
                    st = dict(st, post_prune_error=repr(e)[:300])
"""
s = s.replace(anchor, add + anchor)
p.write_text(s)
base = json.load(open('/workspace/p56stage/p5_config.json'))
e = dict(base, variant='P7e'); e.pop('prune_min_len'); e['post_prune'] = 2
json.dump(e, open('/workspace/p56stage/p7e_config.json', 'w'), indent=1)
print(json.dumps(e))
