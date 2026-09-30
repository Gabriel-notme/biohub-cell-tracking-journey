def make_baseline_streaming(source):
    marker='# Write one complete test submission using the active post-processing configuration'
    assert source.count(marker)==1
    launch='''# Reference prediction has finished using both GPUs. Its postprocessing uses GPU 0.
_stream_log=Path('/kaggle/working/lineage_stream.log').open('w')
_stream_command=[sys.executable,'-u',str(LINEAGE_ROOT/'streaming_pipeline.py'),
    '--artifact',str(LINEAGE_ROOT),'--selection',str(LINEAGE_ROOT/'final_selection.json'),
    '--variant',VARIANT,'--data',str(TEST_DIR),'--graphs','/kaggle/working/reference_graphs','--work','/kaggle/working']
_stream_environment={**os.environ,'BIOHUB_BASE_REPO':str(REPO_DIR)}
_lineage_stream_proc=subprocess.Popen(_stream_command,env=_stream_environment,stdout=_stream_log,stderr=subprocess.STDOUT)
print('Started exact streaming lineage on the available second GPU')

'''
    source=source.replace(marker,launch+marker)
    original="(_gc/(dataset+'.json')).write_text(json.dumps({'nodes':nodes_by_id,'edges':edges,'stats':filter_stats}))"
    replacement="_graph_tmp=_gc/(dataset+'.reference.tmp');_graph_tmp.write_text(json.dumps({'nodes':nodes_by_id,'edges':edges,'stats':filter_stats}));os.replace(_graph_tmp,_gc/(dataset+'.json'))"
    assert source.count(original)==1
    return source.replace(original,replacement)

STREAM_POSTLUDE='''Path('/kaggle/working/reference_postprocessing_complete.json').write_text(json.dumps({'complete':True}))
while _lineage_stream_proc.poll() is None:
    print('Streaming lineage:',len(list(Path('/kaggle/working/lineage_graphs').glob('*.json'))),'movies complete')
    time.sleep(15)
_stream_log.close()
if _lineage_stream_proc.returncode:
    print(Path('/kaggle/working/lineage_stream.log').read_text()[-16000:])
    raise RuntimeError('Streaming lineage failed')
'''
