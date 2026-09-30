from pathlib import Path
import json
R=Path('/workspace/biohub');names=json.loads((R/'track_data/split.json').read_text())['calibration'];rows=[]
for arm in ['endpoint_only98','endpoint_only90']:
    for name in names:
        paths=[R/'track_evaluation'/label/arm/'graphs'/(name+'.json') for label in ['TrackSetDual_cal','TrackSetDual_sparse_cal']]
        if not all(p.exists() for p in paths):continue
        a,b=[json.loads(p.read_text()) for p in paths]
        edges=lambda g:{(int(e['source_id']),int(e['target_id'])) for e in g['edges']}
        aa,bb=edges(a),edges(b);base=json.loads((R/'reproduced_B4'/(name+'.json')).read_text());original=edges(base)
        assert original.issubset(bb),'Endpoint-only sparse path removed inherited edges'
        assert a['nodes']==b['nodes']==base['nodes'],'Node coordinates changed'
        rows.append({'movie':name,'arm':arm,'dense_added':len(aa-original),'sparse_added':len(bb-original),'edge_difference':len(aa^bb)})
result={'comparisons':len(rows),'total_edge_difference':sum(r['edge_difference'] for r in rows),'inherited_edges_and_nodes_preserved':True,'rows':rows}
(R/'sparse_set_comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
