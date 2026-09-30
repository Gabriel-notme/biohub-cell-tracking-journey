"""Add 'dense_relink' (+ optional second free-end-linking pass) to p_stage4.py, after post_prune."""
from pathlib import Path
p = Path('/workspace/p56stage/p_stage4.py'); s = p.read_text()
if "cfg.get('dense_relink')" in s:
    print('already patched'); raise SystemExit
anchor = "            if cfg.get('dc_pass2'):\n"
assert s.count(anchor) == 1
add = """            if cfg.get('dense_relink') is not None:
                try:
                    import dense_relink
                    dr = cfg['dense_relink']
                    ddir = Path(os.environ.get('DENSE_DIR') or dr.get('dir') or (str(Path(a.full).parent / 'dense') if a.full else '.'))
                    n11, e11, st11 = dense_relink.apply(nodes2, ne, ddir / (name + '.npz'), fullp, Path(a.config).parent / dr.get('model', 'dense_lgb.json'), th=float(dr['th']))
                    check_graph(n11, e11, nodes)
                    nodes2, ne = n11, e11; st = dict(st, **st11)
                    if dr.get('el2') and cfg.get('edge_link') is not None and fullp is not None and fullp.exists():
                        import edge_link, prune
                        el = cfg['edge_link']
                        n12, e12, st12 = edge_link.apply(nodes2, ne, fullp, Path(a.config).parent / el.get('model', 'edge_lgb.json'), th=float(el['th']), allow_gap2=bool(el.get('gap2', True)))
                        n12, e12, _ = prune.prune_fragments(n12, e12, int(cfg.get('post_prune') or 2))
                        check_graph(n12, e12, nodes)
                        nodes2, ne = n12, e12; st = dict(st, el2_links=st12.get('el_gap1', 0) + st12.get('el_gap2', 0))
                except Exception as e:
                    st = dict(st, dense_relink_error=repr(e)[:300])
"""
s = s.replace(anchor, add + anchor)
p.write_text(s)
print('patched')
