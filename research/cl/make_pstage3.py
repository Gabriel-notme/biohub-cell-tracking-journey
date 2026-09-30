"""Build p_stage3.py from p_stage2.py: adds learned relink + learned free-end linking after the P3 steps."""
import json, shutil
from pathlib import Path
SRC = Path('/workspace/p12ds/p_stage2.py'); DST = Path('/workspace/p56stage'); DST.mkdir(exist_ok=True)
s = SRC.read_text()
anchor = "                nodes2, ne, pst = prune.prune_fragments(nodes2, ne, int(cfg['prune_min_len'])); st = dict(st, **pst)\n"
assert s.count(anchor) == 1
add = anchor + """            cdir = Path(a.config).parent
            if cfg.get('relink') is not None and fullp is not None and fullp.exists():
                try:
                    import relink, edge_link
                    n5, e5, st5 = relink.apply(nodes2, ne, edge_link.load_full(fullp), cdir / cfg['relink'].get('model', 'relink_lgb.json'), th=float(cfg['relink']['th']))
                    check_graph(n5, e5, nodes)
                    nodes2, ne = n5, e5; st = dict(st, **st5)
                except Exception as e:
                    st = dict(st, relink_error=repr(e)[:300])
            elif cfg.get('relink') is not None:
                st = dict(st, relink_skipped='no_full_graph')
            if cfg.get('edge_link') is not None and fullp is not None and fullp.exists():
                try:
                    import edge_link
                    el = cfg['edge_link']
                    n6, e6, st6 = edge_link.apply(nodes2, ne, fullp, cdir / el.get('model', 'edge_lgb.json'), th=float(el['th']), allow_gap2=bool(el.get('gap2', True)))
                    check_graph(n6, e6, nodes)
                    nodes2, ne = n6, e6; st = dict(st, **st6)
                except Exception as e:
                    st = dict(st, edge_link_error=repr(e)[:300])
            elif cfg.get('edge_link') is not None:
                st = dict(st, edge_link_skipped='no_full_graph')
"""
s = s.replace(anchor, add)
old_rep = "'dsr_errors': sum(1 for r in records if 'dsr_error' in r['pstage'] or 'dfork_error' in r['pstage'])}"
assert s.count(old_rep) == 1
new_rep = ("'dsr_errors': sum(1 for r in records if 'dsr_error' in r['pstage'] or 'dfork_error' in r['pstage']), "
           "'relinked': sum(r['pstage'].get('rl_applied', 0) for r in records), 'edge_linked': sum(r['pstage'].get('el_gap1', 0) + r['pstage'].get('el_gap2', 0) for r in records), "
           "'link_errors': sum(1 for r in records if 'relink_error' in r['pstage'] or 'edge_link_error' in r['pstage'])}")
s = s.replace(old_rep, new_rep)
s = s.replace('"""P-series post-lineage stage: division completion on the final lineage graph.',
              '"""P-series post-lineage stage (v3): division completion on the final lineage graph, then learned relinking and\nlearned free-end linking driven by the saved pre-ILP candidate graph.', 1)
(DST / 'p_stage3.py').write_text(s)
for f in ['relink.py', 'edge_link.py', 'lgb_np.py', 'edge_lgb.json', 'relink_lgb.json']:
    shutil.copy('/workspace/cl/' + f, DST / f)
for f in Path('/workspace/p12ds').glob('*.py'):
    if not (DST / f.name).exists(): shutil.copy(f, DST / f.name)
p3 = json.load(open('/workspace/p12ds/p3_config.json'))
p4 = json.load(open('/workspace/p12ds/p4_config.json'))
c5 = dict(p3, variant='P5', relink={'th': 0.65, 'model': 'relink_lgb.json'}, edge_link={'th': 0.4, 'gap2': True, 'model': 'edge_lgb.json'})
json.dump(c5, open(DST / 'p5_config.json', 'w'), indent=1)
print('built', sorted(p.name for p in DST.iterdir()))
print(json.dumps(c5))
