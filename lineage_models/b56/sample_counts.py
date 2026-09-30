from pathlib import Path
import json
R=Path('/workspace/biohub')
for folder in ['track_predicted3','track_b4w_3']:
    rows=json.loads((R/folder/'preparation.json').read_text());out={}
    for g in ['train','calibration']:
        q=[r for r in rows if r.get('group')==g];out[g]={k:sum(r[k] for r in q) for k in ['edges','forks','edge_positive','fork_positive']};out[g]['movies']=len(q)
    print(folder,json.dumps(out))
