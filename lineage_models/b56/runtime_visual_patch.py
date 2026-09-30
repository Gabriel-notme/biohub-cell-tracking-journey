"""Equivalent elimination of unused visual-edge candidate predictions."""
from pathlib import Path

def make_visual_sparse(source):
    source=source.replace('from visual_motion_features import visual_edge_features','from visual_motion_features import visual_edge_features\nfrom candidate_rows import candidate_rows')
    old="self.appearance.config=self.config;prior=self.appearance.predict(name,nodes,edges);graphsha="
    new="""self.appearance.config=self.config
        edges_only=self.config['fork_threshold']>1 and self.config['fork_veto']==0 and self.config['preserve_inherited_divisions'] and not self.config.get('fork_repair',False)
        if edges_only:
            er,fr=candidate_rows(nodes,edges,self.config,include_forks=False)
            prior={'edge':er,'fork':fr,'fork_logits':np.empty((1,0),np.float32)}
        else:
            prior=self.appearance.predict(name,nodes,edges)
        graphsha="""
    assert source.count(old)==1
    source=source.replace(old,new).replace("'visual_edge_'","'visual_edge_sparse_v1_'")
    compile(source,'visual_edge_refine.py','exec')
    return source

def notebook_patch(visual=True,early_stream=False):
    sources={}
    if visual:sources['visual_edge_refine.py']=make_visual_sparse((Path(__file__).parent/'visual_edge_refine.py').read_text())
    if early_stream:
        source=(Path(__file__).parent/'streaming_pipeline.py').read_text()
        assert source.count('    launch(1)\n')==1
        source=source.replace('    launch(1)\n','    launch(1)\n    launch(0)\n')
        source=source.replace('if 0 in processes and all(p.poll()==0 for p in processes.values()):break','if barrier.exists() and 0 in processes and all(p.poll()==0 for p in processes.values()):break')
        source=source.replace('        export_submission(a,started)',"        assert {p.stem for p in Path(a.data).glob('*.zarr')}=={p.stem for p in Path(a.graphs).glob('*.json')}\n        export_submission(a,started)")
        compile(source,'streaming_pipeline.py','exec');sources['streaming_pipeline.py']=source
    return '''
# Keep the authenticated model artifact immutable; install an explicit, logged
# runtime-only source override in a cloud-local copy. All weights remain linked.
import shutil
_original_lineage_root=LINEAGE_ROOT
_runtime_lineage_root=Path('/kaggle/working/lineage_runtime_sources')
_runtime_lineage_root.mkdir(exist_ok=True)
for _p in LINEAGE_ROOT.iterdir():
    if not _p.is_file():continue
    _dst=_runtime_lineage_root/_p.name
    if _p.suffix=='.py':shutil.copyfile(_p,_dst)
    elif not _dst.exists():_dst.symlink_to(_p)
_overrides=%r
for _name,_source in _overrides.items():(_runtime_lineage_root/_name).write_text(_source)
LINEAGE_ROOT=_runtime_lineage_root
RUNTIME_SOURCE_OVERRIDES={_name:hashlib.sha256(_source.encode()).hexdigest() for _name,_source in _overrides.items()}
print('Equivalent runtime source overrides:',RUNTIME_SOURCE_OVERRIDES)
'''%sources
