"""Generate p_stage10.py (P16) from the deployed p_stage9.py: identical, plus the b1 seed-ensemble division-completion scorer
when the config has a 'b1ens' entry (model files are resolved next to the stage script)."""
import sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
s = src.read_text()
old = "    cfg = load_cfg(a)\n"
new = old + ("    if cfg.get('b1ens'):\n"
             "        import b1ens_dc\n"
             "        b1ens_dc.install(dc, [Path(__file__).parent / f for f in cfg['b1ens']['models']], K=int(cfg['b1ens'].get('K', 300)))\n")
assert s.count(old) == 1, s.count(old)
i = s.index(old); assert 'import div_complete as dc' in s[:i] and s[:i].rfind('def worker') > s[:i].rfind('def load_cfg'), 'unexpected layout'
dst.write_text(s.replace(old, new))
print('wrote', dst)
