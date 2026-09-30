from pathlib import Path
import json
r=json.loads(Path('/workspace/biohub/track_residual3/importance_protocol.json').read_text())
result={k:{f:sum(v[k][f] for v in r['movies']) for f in ['wrong_inherited','missed_truth','examples']} for k in ['edge','fork']}
print(json.dumps(result,indent=2))
