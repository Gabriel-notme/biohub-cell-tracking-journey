import json, glob
from collections import Counter
D = [json.load(open(f)) for f in sorted(glob.glob('/workspace/cl/p16/check/ablate/detail/*.json'))]
print('movies', len(D))
print('same_apply', sum(d['same_apply'] for d in D), 'same_p17', sum(d['same_p17'] for d in D), 'coord_same', sum(d['coord_same'] for d in D))
for d in D:
    if not d['same_p17']: print('  MISMATCH', d['movie'], d['set'], 'n15', d['n15'], 'n17', d['n17'], len(d['rm_cd']) + len(d['rm_ff']) + len(d['rm_sb']), d['ps17'])
    if any('error' in k for k in d['ps17']): print('  P17ERR', d['movie'], d['ps17'])
tot = Counter()
for d in D:
    tot['cd'] += len(d['rm_cd']); tot['ff'] += len(d['rm_ff']); tot['sb'] += len(d['rm_sb']); tot['sbforks'] += len(d['sb']); tot['forks15'] += d['nf15']
    for k, v in d['matched_rm'].items(): tot['matched_' + k] += v
    tot['mov_sb'] += bool(d['sb']); tot['mov_cd'] += bool(d['rm_cd']); tot['mov_ff'] += bool(d['rm_ff'])
print('removed nodes', dict(tot))
sb = [dict(s, movie=d['movie'], set=d['set']) for d in D for s in d['sb']]
print('sb forks total', len(sb), '| evaluable in P15 (tp or fp fork):', sum(s['tp15'] or s['fp15'] for s in sb), '| TP forks:', sum(s['tp15'] for s in sb),
      '| FP forks:', sum(s['fp15'] for s in sb), '| parent matched to GT node:', sum(s['g'] is not None for s in sb),
      '| parent matched GT node is a GT divider:', sum(s['g_outdeg'] == 2 for s in sb), '| GT divider within 10um, dt in -1..1:', sum(bool(s['near_gtdiv']) for s in sb),
      '| still a fork in P17:', sum(s['fork17'] for s in sb))
print('sb dropped-branch length hist', Counter(len(s['drop']) for s in sb))
print('sb forks with any dropped node matched to GT', sum(any(x is not None for x in s['drop_matched']) for s in sb))
for s in sb:
    if s['tp15'] or s['fp15'] or s['g'] is not None or s['near_gtdiv'] or any(x is not None for x in s['drop_matched']):
        print('  SB', s['movie'], s['set'], 'p', s['parent'], 't', s['t'], 'drop', s['drop'], 'keep', s['keep'], 'g', s['g'], 'gout', s['g_outdeg'], 'tp15', s['tp15'], 'fp15', s['fp15'],
              'near', s['near_gtdiv'], 'dropm', s['drop_matched'], 'keepm', s['keep_matched'])
print('--- per-movie evaluable changes')
for d in D:
    if d['tp_lost'] or d['tp_gained'] or d['fp_removed'] or d['fp_added'] or d['fpf_removed'] or d['fpf_added'] or d['tpf_removed'] or d['tpf_added']:
        print(d['movie'], d['set'], 'TPlost', d['tp_lost'], 'TPgain', d['tp_gained'], 'FPrem', d['fp_removed'], 'FPadd', d['fp_added'],
              'fpf-', d['fpf_removed'], 'fpf+', d['fpf_added'], 'tpf-', d['tpf_removed'], 'tpf+', d['tpf_added'], 'c15', d['cnt15'], 'c17', d['cnt17'])
