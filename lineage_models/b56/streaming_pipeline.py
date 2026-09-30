"""Overlap reference postprocessing on GPU 0 with exact lineage work on GPU 1."""
from pathlib import Path
import os,json,time,argparse,hashlib,subprocess,sys,fcntl

def worker(a):
    from lineage_pipeline import LineagePipeline
    import numpy as np,torch
    assert torch.cuda.is_available()
    root=Path(a.work);graphs=Path(a.graphs);dest=root/'lineage_graphs';dest.mkdir(exist_ok=True)
    claims=root/'lineage_claims';claims.mkdir(exist_ok=True)
    names=sorted(p.stem for p in Path(a.data).glob('*.zarr'))
    selection=json.loads(Path(a.selection).read_text())[a.variant]
    pipeline=LineagePipeline(a.artifact,selection,a.data,root/('lineage_cache_'+str(a.worker)))
    identity=hashlib.sha256(json.dumps(pipeline.identity,sort_keys=True).encode()).hexdigest()
    completed=set()
    while len(completed)<len(names):
        available=[n for n in names if n not in completed and (graphs/(n+'.json')).exists()]
        available.sort(key=lambda n:(graphs/(n+'.json')).stat().st_size,reverse=True)
        worked=False
        for name in available:
            with (claims/(name+'.lock')).open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                source=graphs/(name+'.json');raw=source.read_bytes();input_sha=hashlib.sha256(raw).hexdigest();target=dest/(name+'.json')
                if target.exists():
                    saved=json.loads(target.read_text())
                    if saved.get('identity')==identity and saved.get('input_sha256')==input_sha:
                        completed.add(name);continue
                d=json.loads(raw);nodes={int(k):v for k,v in d['nodes'].items()}
                nn,ee,stats=pipeline.refine(name,nodes,d['edges'])
                assert all(np.isfinite([v[k] for k in ['z','y','x']]).all() for v in nn.values())
                result={'nodes':nn,'edges':ee,'stats':stats,'identity':identity,'input_sha256':input_sha}
                tmp=target.with_name(name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(result));os.replace(tmp,target)
                print('STREAM_MOVIE_COMPLETE',a.worker,name,json.dumps(stats),flush=True)
                for p in pipeline.cache.rglob('*'):
                    if p.is_file() and name in p.name and p.suffix in ['.npz','.json']:p.unlink()
                completed.add(name);worked=True
                break
        if not worked and len(completed)<len(names):time.sleep(2)

def main(a):
    import torch
    from parallel_pipeline import export_submission
    assert torch.cuda.device_count()>=2
    root=Path(a.work);root.mkdir(parents=True,exist_ok=True)
    barrier=root/'reference_postprocessing_complete.json'
    processes={};logs={};started=time.time()
    def launch(gpu):
        logs[gpu]=root/('lineage_stream_worker_'+str(gpu)+'.log')
        cmd=[sys.executable,'-u',__file__,'--worker',str(gpu),'--artifact',a.artifact,'--selection',a.selection,'--variant',a.variant,'--data',a.data,'--graphs',a.graphs,'--work',a.work]
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','POLARS_MAX_THREADS':'2'}
        processes[gpu]=subprocess.Popen(cmd,env=env,stdout=logs[gpu].open('w'),stderr=subprocess.STDOUT)
    launch(1)
    try:
        while True:
            failed=[gpu for gpu,p in processes.items() if p.poll() not in [None,0]]
            if failed:
                for gpu in failed:print(logs[gpu].read_text()[-16000:],flush=True)
                raise RuntimeError('Streaming lineage worker failed')
            if barrier.exists() and 0 not in processes:
                expected={p.stem for p in Path(a.data).glob('*.zarr')};found={p.stem for p in Path(a.graphs).glob('*.json')}
                assert expected==found,expected^found
                launch(0)
            if 0 in processes and all(p.poll()==0 for p in processes.values()):break
            done=len(list((root/'lineage_graphs').glob('*.json')))
            print('STREAM_PROGRESS',done,'minutes',round((time.time()-started)/60,1),'reference_complete',barrier.exists(),flush=True)
            time.sleep(10)
        export_submission(a,started)
    finally:
        for p in processes.values():
            if p.poll() is None:p.terminate()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int)
    for name in ['artifact','selection','variant','data','graphs','work']:p.add_argument('--'+name,required=True)
    a=p.parse_args();worker(a) if a.worker is not None else main(a)
