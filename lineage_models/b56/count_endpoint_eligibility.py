from pathlib import Path
import json
from endpoint_eligibility import possible_endpoint_pairs
from evaluate_track_stage import CONFIGS
R=Path('/workspace/biohub');names=json.loads((R/'track_data/split.json').read_text())['calibration'];result={}
for name in names:
    g=json.loads((R/'reproduced_B4'/(name+'.json')).read_text());result[name]=len(possible_endpoint_pairs({int(k):v for k,v in g['nodes'].items()},g['edges'],CONFIGS['endpoint_only90']))
print(json.dumps({'movies':len(result),'zero_eligible':sum(v==0 for v in result.values()),'pairs':sum(result.values()),'counts':result},indent=2))
