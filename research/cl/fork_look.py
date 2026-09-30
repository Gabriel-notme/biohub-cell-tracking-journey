import json
R = json.load(open('/workspace/cl/forks_p3_hold36.json'))
import numpy as np
keys = ['t', 'hist', 'la', 'lb', 'ea', 'eb', 'd_pa', 'd_pb', 'd_ab']
for lab in ['TP', 'FP']:
    for r in sorted([r for r in R if r['lab'] == lab], key=lambda r: r['src']):
        print(lab, r['src'], r['movie'], ' '.join('%s=%s' % (k, round(r[k], 1) if isinstance(r[k], float) else r[k]) for k in keys))
U = [r for r in R if r['lab'] == 'U']
for k in keys:
    print('U', k, np.percentile([r[k] for r in U], [10, 25, 50, 75, 90]).round(1))
