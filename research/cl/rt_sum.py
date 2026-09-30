import json, numpy as np
rows = json.load(open('/workspace/cl/p16/redteam_p20/rt_a.json'))
print('movies', len(rows))
fn = np.array([r['nodes_rm'] / r['n15'] for r in rows]); fe = np.array([r['edges_rm'] / max(1, r['e15']) for r in rows])
print('nodes_add total', sum(r['nodes_add'] for r in rows), 'edges_add total', sum(r['edges_add'] for r in rows), 'coord_changed', sum(r['coord_changed'] for r in rows))
N = sum(r['n15'] for r in rows); DN = sum(r['nodes_rm'] for r in rows)
print('total nodes deleted %d / %d (%.4f%%); edges deleted %d / %d' % (DN, N, 100 * DN / N, sum(r['edges_rm'] for r in rows), sum(r['e15'] for r in rows)))
for nm, a in (('node', fn), ('edge', fe)):
    print(nm, 'frac deleted: max %.4f%% p99 %.4f%% p95 %.4f%% median %.4f%% mean %.4f%%' % tuple(100 * x for x in (a.max(), np.percentile(a, 99), np.percentile(a, 95), np.median(a), a.mean())))
print('movies >2% nodes:', int((fn > 0.02).sum()), ' >1%:', int((fn > 0.01).sum()), ' >0.5%:', int((fn > 0.005).sum()))
rules = ['cutdup', 'forkfrag', 'start_trim', 'term_trim', 'par_dup', 'border']
tot = {k: sum(r['by_rule'][k] for r in rows) for k in rules}; print('by rule totals', tot)
for k in rules:
    a = np.array([r['by_rule'][k] / r['n15'] for r in rows]); i = int(a.argmax())
    print('  %-10s max frac %.4f%% (%s %s n=%d del=%d) p99 %.4f%%' % (k, 100 * a.max(), rows[i]['set'], rows[i]['movie'], rows[i]['n15'], rows[i]['by_rule'][k], 100 * np.percentile(a, 99)))
o = np.argsort(-fn)[:8]
print('top movies by node frac:')
for i in o:
    r = rows[i]; print('  %s %s n15=%d del=%d (%.3f%%) e_del=%d (%.3f%%) %s' % (r['set'], r['movie'], r['n15'], r['nodes_rm'], 100 * fn[i], r['edges_rm'], 100 * fe[i], r['by_rule']))
print('forks P15 total', sum(r['forks15'] for r in rows), 'P19r', sum(r['forks19'] for r in rows), 'lost', sum(len(r['forks_lost']) for r in rows), 'new', sum(len(r['forks_new']) for r in rows))
print('fork_break in replay', sum(len(r['fork_break_rule']) for r in rows))
print('fork gap daughters (P15)', sum(r['fork_gap_daughters'] for r in rows))
print('replay_equal', sum(r['replay_equal'] for r in rows), '/', len(rows))
print('invalid19', [(r['movie'], r['valid19']) for r in rows if r['valid19']], 'invalid15', [(r['movie'], r['valid15']) for r in rows if r['valid15']])
errs = [(r['movie'], r['st19']) for r in rows if any('error' in k or 'skip' in k for k in r['st19'])]; print('errors in p19r stats', len(errs), errs[:5])
print('stat mismatch', [(r['movie'], r['st19'], r['by_rule']) for r in rows if r['st19'].get('p17_cd') != r['by_rule']['cutdup'] or r['st19'].get('p17_ff') != r['by_rule']['forkfrag'] or r['st19'].get('p19_removed') != sum(r['by_rule'][k] for k in rules[2:])][:5])
shapes = {}
for r in rows: shapes[tuple(r['shape'])] = shapes.get(tuple(r['shape']), 0) + 1
print('shapes', shapes)
print('prefix', {p: sum(1 for r in rows if r['movie'].startswith(p)) for p in set(r['movie'][:4] for r in rows)})
t = np.array([r['time_total'] for r in rows]); n = np.array([r['n15'] for r in rows])
print('time total %.1fs, max %.2fs, mean %.3fs' % (t.sum(), t.max(), t.mean()))
for i in np.argsort(-n)[:6]:
    r = rows[i]; print('  %s n=%d t=%.2fs %s p15_stage_sec=%s' % (r['movie'], r['n15'], r['time_total'], r['time'], r['st15_sec']))
print('s per 10k nodes %.3f' % (t.sum() / n.sum() * 1e4))
print('p15 stage seconds total %.0f' % sum(r['st15_sec'] or 0 for r in rows))
sm = np.array([r['small_comp_nodes'] / r['n15'] for r in rows]); print('P15 small-comp(<6,no fork) node frac max %.3f%% median %.3f%%' % (100 * sm.max(), 100 * np.median(sm)))
sy = np.array([r['syn_nodes'] / r['n15'] for r in rows]); print('P15 gap_synthetic node frac max %.3f%% median %.3f%%; gap edges total %d' % (100 * sy.max(), 100 * np.median(sy), sum(r['gap_edges'] for r in rows)))
print('nodes per movie: max %d median %d' % (n.max(), np.median(n)))
