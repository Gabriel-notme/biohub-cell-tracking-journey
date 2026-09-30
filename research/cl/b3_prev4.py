"""Run the B3 lineage stack (= B5 selection truncated to its first four stages; identical config and weights) on prev4."""
import os, sys, json, copy
from pathlib import Path
ART = '/workspace/art_b56/artifact_bundle'
RUNW = '/workspace/runs/b5f_prev4/working'
os.environ['BIOHUB_ART'] = ART; os.environ['BIOHUB_BASE_REPO'] = RUNW + '/tracking_repo'
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, ART)
from lineage_pipeline import LineagePipeline
sel = copy.deepcopy(json.load(open(ART + '/final_selection.json'))['B5']); sel['stages'] = sel['stages'][:4]
p = LineagePipeline(ART, sel, '/workspace/data/train', '/workspace/cl/cache_b3prev4')
out = Path('/workspace/cl/b3_prev4'); out.mkdir(exist_ok=True)
for name in [l.strip() for l in open('/workspace/preview4.txt') if l.strip()]:
    d = json.load(open(RUNW + '/reference_graphs/%s.json' % name))
    nodes = {int(k): v for k, v in d['nodes'].items()}
    n2, e2, st = p.refine(name, nodes, d['edges'])
    (out / (name + '.json')).write_text(json.dumps({'nodes': {str(k): v for k, v in n2.items()}, 'edges': e2}))
    print('B3', name, len(n2), len(e2), flush=True)
# sanity: the same code with all six stages must reproduce local B5 prev4 lineage graphs
sel5 = json.load(open(ART + '/final_selection.json'))['B5']
p5 = LineagePipeline(ART, sel5, '/workspace/data/train', '/workspace/cl/cache_b5prev4')
name = [l.strip() for l in open('/workspace/preview4.txt') if l.strip()][0]
d = json.load(open(RUNW + '/reference_graphs/%s.json' % name)); nodes = {int(k): v for k, v in d['nodes'].items()}
n5, e5, _ = p5.refine(name, nodes, d['edges'])
ref = json.load(open(RUNW + '/lineage_graphs/%s.json' % name))
print('B5 reproduce check', name, {(int(e['source_id']), int(e['target_id'])) for e in e5} == {(int(e['source_id']), int(e['target_id'])) for e in ref['edges']})
