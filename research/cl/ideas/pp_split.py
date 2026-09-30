"""Split /workspace/cl/div_rows.json (b1 fork probs of div_complete candidates on B5 graphs, all rows with fork>=0.3 kept)
into small per-movie candidate files for a CPU replica of the P13 stage."""
import json, os
from collections import defaultdict
OUT = '/workspace/cl/ideas/pp_cands'
os.makedirs(OUT, exist_ok=True)
d = json.load(open('/workspace/cl/div_rows.json'))
by = defaultdict(list)
for r in d:
    if r['fork'] >= 0.5:
        by[r['movie']].append({k: r.get(k) for k in ['p', 'a', 'b', 'q', 'typ', 'fork', 'e_old']})
movies = {r['movie'] for r in d}
for m in movies:
    json.dump(by.get(m, []), open(os.path.join(OUT, m + '.json'), 'w'))
print('movies', len(movies), 'rows', sum(len(v) for v in by.values()))
