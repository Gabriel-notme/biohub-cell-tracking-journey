"""check2/holdout (b): count distinct evaluated variants on the 199 movies from the inventory (dedup by per-movie count signature)."""
import json, re
import numpy as np
from collections import defaultdict
inv = json.load(open('/workspace/cl/p16/check2/holdout/inventory.json'))


def group(o):
    f = o['file']
    if 'chk2_' in f: return None  # other check2 agents (this session's checks)
    if o['mtime'].startswith('09-26'):
        if 'r4_' in f or 'r5_' in f: return 'r4/r5 (09-26 evening, P15 base)'
        return 'r2/r3/combo (09-26, P13/P14 base)'
    if '/check/' in f: return 'P17 multi-check (09-28, P15 base)'
    if 'p16_divnet' in f or 'p16_linknet' in f: return 'p16 divnet/linknet (09-28)'
    if 'p17_rules' in f or 'rows_p17' in f or 'rows_p18' in f: return 'p17/p18 (09-28)'
    if 'p19r' in f or 'p19s' in f or 'p19small' in f: return 'p19r/p19s assembly (09-28)'
    if '/p19/' in f or 'p19' in f: return 'p19 family agents (09-28, P17 base)'
    return 'other'


G = defaultdict(list); seen = set()
for o in inv:
    g = group(o)
    if g is None: continue
    key = (o['base'], round(o['d'], 9), o['dnodes'], o['dtp'], o['dfp'], o['ddtp'], o['ddfp'])  # dedupe across files
    if o['dup_of'] or key in seen: continue
    seen.add(key); G[g].append(o)
tot = 0
for g, os_ in sorted(G.items()):
    ds = np.array([o['d'] for o in os_]); sd = np.array([o['sd'] for o in os_]); z = ds / np.maximum(sd, 1e-7)
    tot += len(os_)
    print('%-40s unique variants %3d | delta median %+.5f  range [%+.5f, %+.5f] | #>0 %d  #lo>0 %d | z>2: %d' % (
        g, len(os_), np.median(ds), ds.min(), ds.max(), (ds > 0).sum(), sum(1 for o in os_ if o['lo'] > 0), (z > 2).sum()))
print('total unique variants (excluding check2 agents):', tot)
p19 = G['p19 family agents (09-28, P17 base)']
print('\np19 family agents detail (P17 base):')
for o in sorted(p19, key=lambda o: o['file']):
    print('  %-40s v%d d %+.5f sd %.5f eval %+.5f nodes %+6d tp %+4d fp %+4d' % (o['file'][-40:], o['vi'], o['d'], o['sd'], o['d_eval'], o['dnodes'], o['dtp'], o['dfp']))
