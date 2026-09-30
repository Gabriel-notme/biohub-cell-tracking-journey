from pathlib import Path
import os,time,json,fcntl
R=Path('/workspace/biohub')
label='training';worker=os.environ['TRACK_POST_WORKER'];os.environ['BIOHUB_POST_ONLY']='1'
os.environ.update(BIOHUB_DATA_ROOT=str(R/(label+'_videos')),BIOHUB_WORKDIR=str(R/('baseline_post_live_'+label+worker)),BIOHUB_UNET_BATCH_SIZE='8',BIOHUB_ALLOW_PIP_INSTALL='1',BIOHUB_VALIDATOR_ENABLE='0',POLARS_MAX_THREADS='4',OMP_NUM_THREADS='4')
s=(R/'cloud_baseline.py').read_text()
s=s.replace("if input_root.exists():\n        candidates.extend(input_root.rglob('ARTIFACT_MANIFEST.json'))", "if input_root.exists() and not any(p.is_file() for p in candidates):\n        candidates.extend(input_root.rglob('ARTIFACT_MANIFEST.json'))")
s=s.replace("if input_root.exists():\n        for name in ('checkpoint_last.pt', 'best.pt', 'last.pt'):","if input_root.exists() and not any(p.is_file() for p in candidates):\n        for name in ('checkpoint_last.pt', 'best.pt', 'last.pt'):")
cut=s.index('_inference_resume_env_keys =');post=s.index('import tracksdata as td',cut);end=s.index('\ndef write_test_submission',post)
scope={'__name__':'baseline_post_library'};exec(compile(s[:cut]+s[post:end],'baseline_post_library','exec'),scope)
dst=R/'training_baseline_graphs';dst.mkdir(exist_ok=True);source=R/('baseline_'+label)/'tracking_repo/predictions';split=json.loads((R/'track_data/split.json').read_text());expected=set(split['train' if label=='training' else 'new_audit'])
locks=R/'training_post_claims';locks.mkdir(exist_ok=True)
while True:
    found={p.stem for p in dst.glob('*.json')}
    if expected<=found:break
    worked=False
    for path in sorted(source.glob('*/unet_transformer*/split_0/*.geff')):
        name=path.stem
        if name not in expected or name in found:continue
        with (locks/(name+'.lock')).open('a') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if (dst/(name+'.json')).exists():continue
            recent=max((p.stat().st_mtime for p in path.rglob('*') if p.is_file()),default=time.time())
            if time.time()-recent<30:continue
            try:
                graph=scope['graph_from_geff'](path)
                nodes={int(row['node_id']):{'node_id':int(row['node_id']),'t':int(row['t']),**{k:float(row[k]) for k in ['z','y','x']}} for row in graph.node_attrs().iter_rows(named=True)}
                edges=[{'source_id':int(row['source_id']),'target_id':int(row['target_id']),'edge_prob':None if row.get('edge_prob') is None else float(row['edge_prob'])} for row in graph.edge_attrs().iter_rows(named=True)]
            except Exception as exc:print('WAIT_GRAPH',name,type(exc).__name__,flush=True);continue
            started=time.time();nodes,edges,stats=scope['filter_output_graph'](nodes,edges,dataset=name,deepcenter_bundle=scope['DEEPCENTER_VETO_DETECTOR'])
            tmp=dst/(name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps({'nodes':nodes,'edges':edges,'stats':stats}));os.replace(tmp,dst/(name+'.json'))
            print('STREAM_BASE_READY',name,len(nodes),len(edges),round(time.time()-started),flush=True);worked=True;break
    if not worked:time.sleep(5)
print('ALL_STREAM_GRAPHS_READY',flush=True)
