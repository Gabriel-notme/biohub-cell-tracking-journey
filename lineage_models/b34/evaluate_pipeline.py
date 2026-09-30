from pathlib import Path
import os,json,time,argparse,hashlib,fcntl
os.environ.setdefault('OMP_NUM_THREADS','3');os.environ.setdefault('POLARS_MAX_THREADS','3');os.environ['BIOHUB_BASE_REPO']='/workspace/biohub/baseline_validation/tracking_repo'
from lineage_pipeline import LineagePipeline
from evaluate_events import score_movie,summarise,paired_interval,TRAIN
R=Path('/workspace/biohub')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True);p.add_argument('--variant',required=True);p.add_argument('--label',required=True);p.add_argument('--group',default='calibration',choices=['calibration','audit']);p.add_argument('--limit',type=int);p.add_argument('--input',default='baseline_graphs');a=p.parse_args()
    selection=json.loads(Path(a.selection).read_text())[a.variant]
    if a.group=='audit':assert selection.get('calibration_frozen') is True,'Audit requires a frozen candidate'
    names=json.loads((R/'events/split.json').read_text())[a.group]
    if a.limit:names=names[:a.limit]
    dest=R/'pipeline_evaluation'/a.label;dest.mkdir(parents=True,exist_ok=True);lock=(dest/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pipeline=LineagePipeline(R,selection,TRAIN,R);identity={**pipeline.identity,'group':a.group,'names':names};identitypath=dest/'identity.json'
    if a.input!='baseline_graphs':identity['reference_graphs']=a.input
    if identitypath.exists():assert json.loads(identitypath.read_text())==identity,'Refusing to mix model or configuration versions'
    identitypath.write_text(json.dumps(identity,indent=2));rows=[];base=[]
    for name in names:
        path=dest/(name+'.json');prior=R/'prior_evaluation'/('B2_'+a.group+'_'+name+'.json')
        while not prior.exists():time.sleep(10)
        base.append(json.loads(prior.read_text()))
        if path.exists():row=json.loads(path.read_text())
        else:
            raw=json.loads((R/a.input/(name+'.json')).read_text());nodes={int(k):v for k,v in raw['nodes'].items()};nn,ee,stats=pipeline.refine(name,nodes,raw['edges']);row=score_movie(name,nn,ee);row['refinement']=stats
            path.write_text(json.dumps(row,indent=2));gd=dest/'graphs';gd.mkdir(exist_ok=True);(gd/(name+'.json')).write_text(json.dumps({'nodes':nn,'edges':ee}));print('PIPELINE_SCORED',name,json.dumps(row),flush=True)
        rows.append(row);(dest/'progress.json').write_text(json.dumps({'movies':len(rows),'total':len(names),'baseline':summarise(base),'candidate':summarise(rows)},indent=2))
    result={'label':a.label,'group':a.group,'identity':identity,'baseline':summarise(base),'candidate':summarise(rows),'paired_bootstrap_95':paired_interval(base,rows),'rows':rows}
    result['delta']=result['candidate']['score']-result['baseline']['score'];(dest/'report.json').write_text(json.dumps(result,indent=2));print('PIPELINE_EVALUATION_COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['identity','rows']}),flush=True)
