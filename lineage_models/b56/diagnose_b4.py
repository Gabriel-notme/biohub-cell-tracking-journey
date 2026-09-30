from pathlib import Path
import os,json,time
os.environ['POLARS_MAX_THREADS']='3';os.environ['OMP_NUM_THREADS']='3'
from evaluate_events import *
from prepare_events import graph
from refine_events import structure
from scipy.spatial import cKDTree
from collections import Counter
from tracking_cellmot.division_metrics import score_divisions
R=Path('/workspace/biohub');dest=R/'b4_error_analysis';dest.mkdir(exist_ok=True)
names=json.loads((R/'events/split.json').read_text())['calibration'];pending=set(names);reports=[]
while pending:
    ready=[n for n in names if n in pending and (R/'reproduced_B4'/(n+'.json')).exists()]
    if not ready:time.sleep(15);continue
    for name in ready:
        try:d=json.loads((R/'reproduced_B4'/(name+'.json')).read_text())
        except json.JSONDecodeError:continue
        nodes={int(k):v for k,v in d['nodes'].items()};edges=d['edges'];out,prev,frames,pos=structure(nodes,edges)
        gt,go,gp,gframes,gpos=graph(TRAIN/(name+'.geff'))
        pred=td.graph.InMemoryGraph()
        for k in ['z','y','x']:pred.add_node_attr_key(k,pl.Float64,0.)
        pred.add_node_attr_key('original',pl.Int64,-1)
        ids=list(nodes);assigned=pred.bulk_add_nodes([{'t':int(nodes[n]['t']),'original':n,**{k:float(max(0,int(round(nodes[n][k])))) for k in ['z','y','x']}} for n in ids]);mapping=dict(zip(ids,assigned));reverse=dict(zip(assigned,ids))
        pred.bulk_add_edges([{'source_id':mapping[int(e['source_id'])],'target_id':mapping[int(e['target_id'])]} for e in edges])
        truth=td.graph.IndexedRXGraph.from_geff(TRAIN/(name+'.geff'))
        if isinstance(truth,tuple):truth=truth[0]
        er=evaluate(pred,truth,scale=tuple(SCALE.astype(float)),max_distance=7.)
        mk=td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID;matched={}
        for r in pred.node_attrs().iter_rows(named=True):
            if r.get(mk) is not None and r[mk]>=0:matched.setdefault(int(r[mk]),[]).append(int(r['original']))
        counts=Counter();fn=[]
        for s,ds in go.items():
            for c in ds:
                ps=matched.get(s,[]);pc=matched.get(c,[])
                if any(b in out.get(a,[]) for a in ps for b in pc):counts['recovered']+=1
                elif not ps or not pc:counts['missing_endpoint']+=1
                else:
                    counts['both_detected_wrong_link']+=1
                    if all(not out.get(a) for a in ps):counts['source_has_no_successor']+=1
                    elif any(len(out.get(a,[]))==2 for a in ps):counts['source_is_division']+=1
                    else:counts['source_linked_elsewhere']+=1
                    distances=[float(np.linalg.norm(pos[a]-pos[b])) for a in ps for b in pc]
                    fn.append({'source':int(s),'target':int(c),'t':int(gt[s]['t']),'pred_sources':ps,'pred_targets':pc,'min_step_um':min(distances)})
        dscores=score_divisions(pred,truth,scale=tuple(SCALE.astype(float)),max_distance=7.)
        div=[]
        trees={t:cKDTree([pos[n] for n in ns]) for t,ns in frames.items()}
        for s,ds in go.items():
            if len(ds)!=2:continue
            nearest=[]
            for n in [s,*ds]:
                t=int(gt[n]['t']);dist,i=trees[t].query(gpos[n]);nearest.append({'pred':int(frames[t][i]),'distance':float(dist)})
            div.append({'gt_source':int(s),'t':int(gt[s]['t']),'recovered':int(dscores.scores.get(s,0)),'nearest':nearest})
        record={'movie':name,'counts':dict(counts),'metric':er._asdict(),'gt_nodes':len(gt),'unmatched_gt_nodes':len(set(gt)-set(matched)),'division_recovery':sum(r['recovered'] for r in div),'divisions':len(div)}
        reports.append(record);(dest/(name+'_details.json')).write_text(json.dumps({'wrong_links':fn,'divisions':div},indent=2));(dest/'aggregate.json').write_text(json.dumps({'rows':reports,'counts':dict(sum((Counter(r['counts']) for r in reports),Counter()))},indent=2))
        print('ERROR_AUDIT',json.dumps(record),flush=True);pending.remove(name)
print('ERROR_ANALYSIS_COMPLETE',flush=True)
