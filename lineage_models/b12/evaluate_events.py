from pathlib import Path
import os,sys,json,argparse,time,copy,hashlib
os.environ.setdefault('POLARS_MAX_THREADS','4')
sys.path.insert(0,'/workspace/biohub/official/src')
import numpy as np
import polars as pl
import tracksdata as td
from geff import GeffMetadata
from tracking_cellmot.metrics import evaluate,per_sample_metrics,summarise,node_recall
from refine_events import EventRefiner,DEFAULT_CONFIG
from cell_event import SCALE

ROOT=Path('/workspace/biohub');TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')
METRIC_COMMIT='075fc5f5a52d11077f9dc2b074644618f26939e2'

def to_graph(nodes,edges):
    g=td.graph.InMemoryGraph()
    for k in ['z','y','x']:g.add_node_attr_key(k,pl.Float64,0.)
    old=list(nodes)
    assigned=g.bulk_add_nodes([{'t':int(nodes[n]['t']),**{k:float(max(0,int(round(nodes[n][k])))) for k in ['z','y','x']}} for n in old])
    mapping=dict(zip(old,assigned))
    if edges:g.bulk_add_edges([{'source_id':mapping[int(e['source_id'])],'target_id':mapping[int(e['target_id'])]} for e in edges])
    return g

def score_movie(name,nodes,edges):
    gtpath=TRAIN/(name+'.geff');gt=td.graph.IndexedRXGraph.from_geff(gtpath)
    if isinstance(gt,tuple):gt=gt[0]
    predicted=to_graph(nodes,edges)
    er=evaluate(predicted,gt,scale=tuple(SCALE.astype(float)),max_distance=7.)
    n_total=float((GeffMetadata.read(gtpath).extra or {}).get('estimated_number_of_nodes',float('nan')))
    assert n_total>0,(name,'Missing official estimated node count')
    row=per_sample_metrics(er,n_total,node_recall(predicted,gt));row['movie']=name
    return row

def load_base(name):
    d=json.loads((ROOT/'baseline_graphs'/(name+'.json')).read_text())
    return {int(k):v for k,v in d['nodes'].items()},d['edges']

def paired_interval(base,changed):
    b={r['movie']:r for r in base};c={r['movie']:r for r in changed};names=sorted(b.keys()&c.keys())
    rng=np.random.default_rng(314159);deltas=[]
    for _ in range(1000):
        sample=rng.choice(names,len(names),replace=True)
        deltas.append(summarise([c[n] for n in sample])['score']-summarise([b[n] for n in sample])['score'])
    return np.percentile(deltas,[2.5,50,97.5]).tolist()

def run_arm(refiner,names,label,out):
    rows=[]
    for name in names:
        report=out/(label+'_'+name+'.json')
        if report.exists():rows.append(json.loads(report.read_text()));continue
        nodes,edges=load_base(name)
        stats={}
        if refiner is not None:nodes,edges,stats=refiner.refine(name,nodes,edges)
        row=score_movie(name,nodes,edges);row['refinement']=stats;rows.append(row)
        report.write_text(json.dumps(row,indent=2));print('SCORED',label,name,json.dumps(row),flush=True)
    summary=summarise(rows);(out/(label+'_summary.json')).write_text(json.dumps({'summary':summary,'rows':rows},indent=2))
    print('ARM',label,json.dumps(summary),flush=True)
    return rows,summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',choices=['B1','B2'],required=True);p.add_argument('--audit-only',action='store_true')
    p.add_argument('--models');p.add_argument('--output');p.add_argument('--focus');args=p.parse_args()
    split=json.loads((ROOT/'events/split.json').read_text());out=Path(args.output) if args.output else ROOT/'evaluation'/args.version;out.mkdir(parents=True,exist_ok=True)
    import fcntl
    _evaluation_lock=(out/'evaluation.lock').open('a');fcntl.flock(_evaluation_lock,fcntl.LOCK_EX)
    names=split['calibration'];base_rows,base=run_arm(None,names,'baseline_calibration',out)
    paths=[ROOT/'B1/best.pt'] if args.version=='B1' else [ROOT/'B1/best.pt',ROOT/'B2/best.pt']
    if args.models:paths=[Path(p) for p in args.models.split(',')]
    refiner=EventRefiner(paths,TRAIN,cache_dir=ROOT/'event_cache')
    configs=[
        {'name':'edge_only','edge_weight':.4,'change_penalty':.75,'edge_min_prob':.85,'fork_threshold':1.1},
        {'name':'appearance_cautious','edge_weight':.4,'change_penalty':.75,'edge_min_prob':.85,'fork_threshold':.85},
        {'name':'appearance_balanced','edge_weight':1.,'change_penalty':.5,'edge_min_prob':.70,'fork_threshold':.85},
        {'name':'appearance_strong','edge_weight':2.,'change_penalty':.3,'edge_min_prob':.65,'fork_threshold':.85},
        {'name':'appearance_dominant','edge_weight':8.,'change_penalty':.3,'edge_min_prob':.70,'max_distance':14.,'fork_threshold':.95,'fork_veto_threshold':.10},
        {'name':'appearance_long_motion','edge_weight':4.,'change_penalty':.5,'edge_min_prob':.85,'max_distance':18.,'fork_threshold':.95},
        {'name':'mitosis_cautious','edge_weight':0.,'fork_threshold':.90},
        {'name':'mitosis_strict','edge_weight':0.,'fork_threshold':.98},
        {'name':'mitosis_balanced','edge_weight':0.,'fork_threshold':.75},
        {'name':'joint_mitosis','edge_weight':.4,'change_penalty':.75,'edge_min_prob':.85,'fork_threshold':.70},
        {'name':'division_review_strict','edge_weight':0.,'fork_threshold':.95,'fork_veto_threshold':.03},
        {'name':'division_review_balanced','edge_weight':0.,'fork_threshold':.95,'fork_veto_threshold':.10},
        {'name':'division_review_broad','edge_weight':0.,'fork_threshold':.90,'fork_veto_threshold':.30},
        {'name':'division_review_half','edge_weight':0.,'fork_threshold':.95,'fork_veto_threshold':.50},
        {'name':'division_review_moderate','edge_weight':0.,'fork_threshold':.80,'fork_veto_threshold':.50},
        {'name':'division_review_recall','edge_weight':0.,'fork_threshold':.65,'fork_veto_threshold':.30},
        {'name':'joint_division_review','edge_weight':.4,'change_penalty':.75,'edge_min_prob':.85,'fork_threshold':.90,'fork_veto_threshold':.10},
        {'name':'high_confidence_division_reassignment','edge_weight':0.,'fork_threshold':.99,'fork_veto_threshold':.10,
         'fork_reassign':True,'fork_reassign_old_max':1.01,'fork_reassign_new_min':.5,'fork_from_endpoints':True},
        {'name':'learned_division_reassignment95','edge_weight':0.,'fork_threshold':.95,'fork_veto_threshold':.50,
         'fork_reassign':True,'fork_reassign_old_max':1.01,'fork_reassign_new_min':.5,'fork_from_endpoints':True},
        {'name':'learned_division_reassignment90','edge_weight':0.,'fork_threshold':.90,'fork_veto_threshold':.50,
         'fork_reassign':True,'fork_reassign_old_max':1.01,'fork_reassign_new_min':.5,'fork_from_endpoints':True},
    ]
    if args.version=='B2':
        for c in configs:c.update(ensemble='conservative')
        configs.extend([{'name':'mean_division_review_half','edge_weight':0.,'fork_threshold':.95,'fork_veto_threshold':.50,'ensemble':'mean'},
                        {'name':'mean_division_review_broad','edge_weight':0.,'fork_threshold':.90,'fork_veto_threshold':.30,'ensemble':'mean'}])
        configs.extend([{'name':'temporal_union_strict','edge_weight':0.,'fork_threshold':.99,'fork_veto_threshold':.50,'ensemble':'max_fork'},
                        {'name':'temporal_union_balanced','edge_weight':0.,'fork_threshold':.95,'fork_veto_threshold':.50,'ensemble':'max_fork'}])
        configs.extend([{'name':'specialist_cautious','edge_weight':2.,'change_penalty':.3,'edge_min_prob':.70,'max_distance':14.,
                         'fork_threshold':.95,'fork_veto_threshold':.50,'edge_ensemble':'last','fork_ensemble':'first'},
                        {'name':'specialist_dominant','edge_weight':8.,'change_penalty':.3,'edge_min_prob':.70,'max_distance':14.,
                         'fork_threshold':.95,'fork_veto_threshold':.50,'edge_ensemble':'last','fork_ensemble':'first'}])
        configs.extend([{'name':'division_reassignment','edge_weight':.4,'change_penalty':.75,'edge_min_prob':.85,'fork_threshold':.85,'fork_reassign':True,'fork_from_endpoints':True,'ensemble':'conservative'},
                        {'name':'division_reassignment_strict','edge_weight':0.,'fork_threshold':.95,'fork_reassign':True,'ensemble':'conservative'}])
        configs.extend([{'name':'five_frame_dominant','edge_weight':8.,'change_penalty':.3,'edge_min_prob':.70,'max_distance':14.,'fork_threshold':.95,'fork_veto_threshold':.10,'ensemble':'last'},
                        {'name':'five_frame_division_review','edge_weight':0.,'fork_threshold':.90,'fork_veto_threshold':.30,'ensemble':'last'}])
    if args.focus:configs=[c for c in configs if c['name'] in args.focus.split(',')]
    comparison_path=out/'calibration_comparison.json'
    records=json.loads(comparison_path.read_text()) if comparison_path.exists() else []
    eligible=[r for r in records if r.get('effective_changes',0)>0]
    best=max(eligible,key=lambda r:r['summary']['score']) if eligible else None
    if args.audit_only:
        best=json.loads((out/'selected_config.json').read_text());configs=[]
    for config in configs:
        refiner.config.update(config)
        # Reinitialise omitted settings so one arm cannot contaminate another.
        from refine_events import DEFAULT_CONFIG
        refiner.config={**DEFAULT_CONFIG,**config}
        rows,summary=run_arm(refiner,names,config['name']+'_calibration',out)
        delta=summary['score']-base['score'];interval=paired_interval(base_rows,rows)
        changes=sum(sum(r.get('refinement',{}).get(k,0) for k in ['event_edges_changed','event_divisions_added','event_divisions_removed']) for r in rows)
        record={'config':config,'summary':summary,'delta':delta,'paired_bootstrap_95':interval,'effective_changes':changes}
        records=[r for r in records if r['config']['name']!=config['name']]+[record]
        if changes>0 and (best is None or summary['score']>best['summary']['score']):best=record
        (out/'calibration_comparison.json').write_text(json.dumps(records,indent=2))
    if best is None:raise RuntimeError('No tested setting changed the reference graph; do not relabel an unchanged baseline as a new result.')
    (out/'selected_config.json').write_text(json.dumps(best,indent=2))
    if not args.audit_only:
        print('CALIBRATION_COMPLETE_AUDIT_RESERVED',json.dumps(best),flush=True);raise SystemExit(0)
    audit_names=split['audit'];audit_base_rows,audit_base=run_arm(None,audit_names,'baseline_audit',out)
    refiner.config={**DEFAULT_CONFIG,**best['config']}
    audit_rows,audit=run_arm(refiner,audit_names,'selected_audit',out)
    report={'version':args.version,'metric_commit':METRIC_COMMIT,'scope':'Complete movies, integer CSV-coordinate round trip. New event-head fitting excludes both calibration and audit movies. Public detector/feature pretraining includes these movies, so this is not an unbiased out-of-fold estimate of the entire pipeline.',
        'selected':best,'audit_baseline':audit_base,'audit_model':audit,'audit_delta':audit['score']-audit_base['score'],
        'audit_paired_bootstrap_95':paired_interval(audit_base_rows,audit_rows),'calibration_movies':names,'audit_movies':audit_names,
        'public_leaderboard_score_verified':False,'inference_precision':'float16',
        'model_files':[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
        'calibration_uncertainty':'Exploratory paired bootstrap after configuration selection; not adjusted for selection.'}
    (out/'validation_report.json').write_text(json.dumps(report,indent=2));print('FINAL_VALIDATION',json.dumps(report),flush=True)
