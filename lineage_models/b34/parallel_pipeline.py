"""Shard full lineage inference across two GPUs, then validate and atomically export."""
from pathlib import Path
import os,json,time,argparse,hashlib,subprocess,sys,csv

def worker(a):
    from lineage_pipeline import LineagePipeline
    import torch,numpy as np
    assert torch.cuda.is_available(),'GPU is mandatory'
    root=Path(a.work);graphs=Path(a.graphs);dest=root/'lineage_graphs';dest.mkdir(parents=True,exist_ok=True)
    selection=json.loads(Path(a.selection).read_text())[a.variant];pipeline=LineagePipeline(a.artifact,selection,a.data,root/('lineage_cache_'+str(a.worker)))
    identity=hashlib.sha256(json.dumps(pipeline.identity,sort_keys=True).encode()).hexdigest();names=json.loads((root/('lineage_shard_'+str(a.worker)+'.json')).read_text())
    for name in names:
        source=graphs/(name+'.json');input_sha=hashlib.sha256(source.read_bytes()).hexdigest();target=dest/(name+'.json')
        if target.exists():
            saved=json.loads(target.read_text())
            if saved.get('identity')==identity and saved.get('input_sha256')==input_sha:continue
        d=json.loads(source.read_text());nodes={int(k):v for k,v in d['nodes'].items()};edges=d['edges'];nn,ee,stats=pipeline.refine(name,nodes,edges)
        assert all(np.isfinite([v[k] for k in ['z','y','x']]).all() for v in nn.values())
        result={'nodes':nn,'edges':ee,'stats':stats,'identity':identity,'input_sha256':input_sha};tmp=target.with_name(name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(result));os.replace(tmp,target)
        print('LINEAGE_MOVIE_COMPLETE',name,json.dumps(stats),flush=True)
        # Only temporary feature caches created by this worker are pruned.
        for p in pipeline.cache.rglob('*'):
            if p.is_file() and name in p.name and p.suffix in ['.npz','.json']:p.unlink()
    (root/('lineage_worker_'+str(a.worker)+'_complete.json')).write_text(json.dumps({'identity':identity,'movies':len(names),'gpu':torch.cuda.get_device_name(0)}))

def main(a):
    import torch,zarr
    assert torch.cuda.device_count()>=2,'This frozen deployment requires two GPUs'
    root=Path(a.work);root.mkdir(parents=True,exist_ok=True);data=Path(a.data);graphs=Path(a.graphs)
    names=sorted(p.stem for p in data.glob('*.zarr'));assert names,'No test movies'
    found={p.stem for p in graphs.glob('*.json')};assert found==set(names),(found^set(names))
    # Balance work using source graph sizes; no image or annotation content selects the model.
    shards=[[],[]];sizes=[0,0]
    for name in sorted(names,key=lambda n:(graphs/(n+'.json')).stat().st_size,reverse=True):
        j=min(range(2),key=lambda i:sizes[i]);shards[j].append(name);sizes[j]+=(graphs/(name+'.json')).stat().st_size
    processes=[];started=time.time()
    for gpu,ns in enumerate(shards):
        (root/('lineage_shard_'+str(gpu)+'.json')).write_text(json.dumps(ns));log=(root/('lineage_worker_'+str(gpu)+'.log')).open('w')
        cmd=[sys.executable,'-u',__file__,'--worker',str(gpu),'--artifact',a.artifact,'--selection',a.selection,'--variant',a.variant,'--data',a.data,'--graphs',a.graphs,'--work',a.work]
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','POLARS_MAX_THREADS':'2','MKL_NUM_THREADS':'2'}
        processes.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in processes):
        failed=[i for i,p in enumerate(processes) if p.poll() not in [None,0]]
        if failed:
            for p in processes:
                if p.poll() is None:p.terminate()
            for i in failed:print((root/('lineage_worker_'+str(i)+'.log')).read_text()[-12000:],flush=True)
            raise RuntimeError('Lineage worker failed: '+str(failed))
        done=len(list((root/'lineage_graphs').glob('*.json'))) if (root/'lineage_graphs').exists() else 0
        print('LINEAGE_PROGRESS',done,'/',len(names),'minutes',round((time.time()-started)/60,1),flush=True);time.sleep(15)
    assert all(p.returncode==0 for p in processes)
    export_submission(a,started)

def export_submission(a,started):
    import zarr
    root=Path(a.work);data=Path(a.data);names=sorted(p.stem for p in data.glob('*.zarr'))
    columns=['id','dataset','row_type','node_id','t','z','y','x','source_id','target_id'];output=root/'submission.csv';tmp=output.with_name('submission.lineage.tmp');row_id=0;records=[]
    from collections import Counter
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for name in names:
            d=json.loads((root/'lineage_graphs'/(name+'.json')).read_text());nodes={int(k):v for k,v in d['nodes'].items()};edges=d['edges'];shape=zarr.open_group(str(data/(name+'.zarr')),mode='r')['0'].shape
            assert {int(v['t']) for v in nodes.values()}==set(range(shape[0])),name+' missing frames'
            for n in sorted(nodes):
                v=nodes[n];coords=[max(0,int(round(v[k]))) for k in ['z','y','x']];assert all(c<limit for c,limit in zip(coords,shape[1:])),(name,'coordinate outside image')
                writer.writerow(dict(id=row_id,dataset=name,row_type='node',node_id=n,t=int(v['t']),z=coords[0],y=coords[1],x=coords[2],source_id=-1,target_id=-1));row_id+=1
            pairs=[(int(e['source_id']),int(e['target_id'])) for e in edges];assert len(pairs)==len(set(pairs));assert max(Counter(s for s,d in pairs).values(),default=0)<=2;assert max(Counter(d for s,d in pairs).values(),default=0)<=1
            for s,t in pairs:
                assert s in nodes and t in nodes and nodes[t]['t']==nodes[s]['t']+1
                writer.writerow(dict(id=row_id,dataset=name,row_type='edge',node_id=-1,t=-1,z=-1,y=-1,x=-1,source_id=s,target_id=t));row_id+=1
            records.append({'movie':name,'nodes':len(nodes),'edges':len(edges),'frames':shape[0],**d['stats']})
    os.replace(tmp,output)
    report={'variant':a.variant,'movies':len(names),'rows':row_id,'lineage_seconds':time.time()-started,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'records':records,'leaderboard_score_verified':False,'selection_sha256':hashlib.sha256(Path(a.selection).read_bytes()).hexdigest()}
    (root/'lineage_runtime_validation.json').write_text(json.dumps(report,indent=2));print('LINEAGE_EXPORT_COMPLETE',json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--artifact',required=True);p.add_argument('--selection',required=True);p.add_argument('--variant',required=True);p.add_argument('--data',required=True);p.add_argument('--graphs',required=True);p.add_argument('--work',required=True);a=p.parse_args()
    worker(a) if a.worker is not None else main(a)
