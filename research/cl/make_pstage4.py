"""Build p_stage4.py from p_stage3.py with two optional, config-driven variants:
  pre_links : run learned relink + free-end linking BEFORE division completion (keys 'pre_relink', 'pre_edge_link')
  dc_pass2  : after the link steps, run a second division-completion pass on the repaired graph, then dfork + prune."""
from pathlib import Path
s = Path('/workspace/p56stage/p_stage3.py').read_text()
a1 = "            cands = dc.score_candidates(er, name, nodes, edges)\n"
assert s.count(a1) == 1
pre = """            fullp0 = Path(a.full) / (name + '.geff') if a.full else None
            if (cfg.get('pre_relink') is not None or cfg.get('pre_edge_link') is not None) and fullp0 is not None and fullp0.exists():
                cdir0 = Path(a.config).parent
                try:
                    import relink, edge_link
                    n0, e0 = nodes, edges
                    if cfg.get('pre_relink') is not None:
                        n0, e0, s0 = relink.apply(n0, e0, edge_link.load_full(fullp0), cdir0 / cfg['pre_relink'].get('model', 'relink_lgb.json'), th=float(cfg['pre_relink']['th']))
                        check_graph(n0, e0, nodes); stats['pre_rl'] = s0.get('rl_applied', 0)
                    if cfg.get('pre_edge_link') is not None:
                        el0 = cfg['pre_edge_link']
                        n0, e0, s0 = edge_link.apply(n0, e0, fullp0, cdir0 / el0.get('model', 'edge_lgb.json'), th=float(el0['th']), allow_gap2=bool(el0.get('gap2', True)))
                        check_graph(n0, e0, nodes); stats['pre_el'] = s0.get('el_gap1', 0) + s0.get('el_gap2', 0)
                    nodes, edges = n0, e0
                except Exception as e:
                    stats['pre_link_error'] = repr(e)[:300]
""" + a1
s = s.replace(a1, pre)
a2 = "            elif cfg.get('edge_link') is not None:\n                st = dict(st, edge_link_skipped='no_full_graph')\n"
assert s.count(a2) == 1
p2 = a2 + """            if cfg.get('dc_pass2'):
                try:
                    c2 = dc.score_candidates(er, name, nodes2, ne)
                    e7, st7 = dc.apply(nodes2, ne, c2, **cfg['apply'])
                    import dfork, prune
                    e7, st7b = dfork.resolve(nodes2, e7, **cfg.get('dfork', {'K': 2}))
                    n7, e7, st7c = prune.prune_fragments(nodes2, e7, int(cfg.get('prune_min_len') or 2))
                    check_graph(n7, e7, nodes)
                    nodes2, ne = n7, e7; st = dict(st, dc2_added=st7.get('div_added', 0), dc2_stolen=st7.get('div_stolen', 0), dfork2_removed=st7b.get('dfork_removed', 0))
                except Exception as e:
                    st = dict(st, dc2_error=repr(e)[:300])
"""
s = s.replace(a2, p2)
# stats dict must exist before the pre-link block: it is created as 'stats = {}' right after t0
assert "t0 = time.time(); stats = {}" in s
s = s.replace("stats = dict(st, candidates=len(cands), seconds=round(time.time() - t0, 2))",
              "stats = dict(stats, **st, candidates=len(cands), seconds=round(time.time() - t0, 2))")
Path('/workspace/p56stage/p_stage4.py').write_text(s)
import json
base = json.load(open('/workspace/p56stage/p5_config.json'))
a = dict(base, variant='P7a'); a['pre_relink'] = a.pop('relink'); a['pre_edge_link'] = a.pop('edge_link')
b = dict(base, variant='P7b', dc_pass2=True)
json.dump(a, open('/workspace/p56stage/p7a_config.json', 'w'), indent=1); json.dump(b, open('/workspace/p56stage/p7b_config.json', 'w'), indent=1)
print('ok'); print(json.dumps(a)); print(json.dumps(b))
