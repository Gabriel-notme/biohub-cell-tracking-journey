"""Summarise every rule_eval result file: for each (file, vi) the official-metric delta vs vi=0 on all 199, per embryo, clean40,
B5-clean72 (hold36+prev4+audit32), and the number of sets with a positive delta. Lists variants positive on all of
all/44b6/6bba/clean40/clean72 first."""
import json, glob, sys, os
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
import warnings; warnings.filterwarnings('ignore')
out = []
for f in sorted(glob.glob('/workspace/cl/ideas/rule_eval_last_*.json') + glob.glob('/workspace/cl/rule_eval_last_*.json') + glob.glob('/workspace/cl/nm/*rows*.json')):
    try:
        rows = json.load(open(f))
        if not isinstance(rows, list) or not rows or 'vi' not in rows[0] or 'set' not in rows[0]: continue
    except Exception:
        continue
    base = {r['movie']: r for r in rows if r['vi'] == 0}
    if len(base) < 150: continue
    for vi in sorted({r['vi'] for r in rows} - {0}):
        cur = {r['movie']: r for r in rows if r['vi'] == vi}
        ms = sorted(set(base) & set(cur))
        if len(ms) < 150: continue
        def d(sel):
            sel = [m for m in ms if sel(m)]
            return summarise([cur[m] for m in sel])['score'] - summarise([base[m] for m in sel])['score'] if sel else 0.0
        st = {m: base[m]['set'] for m in ms}
        r = dict(file=os.path.basename(f), vi=vi, all=d(lambda m: True), e44=d(lambda m: m.startswith('44b6')), e6b=d(lambda m: m.startswith('6bba')),
                 clean40=d(lambda m: st[m] in ('hold36', 'prev4')), clean72=d(lambda m: st[m] in ('hold36', 'prev4', 'audit32')),
                 sets_pos=sum(1 for s in ('hold36', 'prev4', 'audit32', 't127a', 't127b') if d(lambda m, s=s: st[m] == s) > 0),
                 nodes=sum(cur[m]['num_pred_nodes'] - base[m]['num_pred_nodes'] for m in ms))
        out.append(r)
out.sort(key=lambda r: -r['all'])
good = [r for r in out if min(r['all'], r['e44'], r['e6b'], r['clean40'], r['clean72']) > 0]
print('POSITIVE ON all/44b6/6bba/clean40/clean72:')
for r in good: print('  %-48s vi=%d all %+.5f 44b6 %+.5f 6bba %+.5f clean40 %+.5f clean72 %+.5f sets+ %d nodes %+d' % (r['file'][:48], r['vi'], r['all'], r['e44'], r['e6b'], r['clean40'], r['clean72'], r['sets_pos'], r['nodes']))
print('TOP 25 BY all:')
for r in out[:25]: print('  %-48s vi=%d all %+.5f 44b6 %+.5f 6bba %+.5f clean40 %+.5f clean72 %+.5f sets+ %d' % (r['file'][:48], r['vi'], r['all'], r['e44'], r['e6b'], r['clean40'], r['clean72'], r['sets_pos']))
