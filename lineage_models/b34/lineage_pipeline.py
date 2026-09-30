"""Frozen, reproducible B3/B4 lineage pipeline shared by validation and deployment."""
from pathlib import Path
import hashlib,json,time
from collections import Counter
import torch
from refine_events import EventRefiner

def refiner_class(kind):
    if kind=='fast_division_review':
        from fast_division_review import FastDivisionReview
        return FastDivisionReview
    if kind=='centroid':
        from centroid_refine import CentroidRefiner
        return CentroidRefiner
    if kind=='coordinate':
        from coordinate_refine import CoordinateRefiner
        return CoordinateRefiner
    if kind=='division_rescue':
        from division_rescue import DivisionRescueRefiner
        return DivisionRescueRefiner
    if kind=='visual_rank':
        from visual_rank_refine import VisualRankRefiner
        return VisualRankRefiner
    if kind=='multiframe_recovery':
        from multiframe_recovery import MultiFrameRecoveryRefiner
        return MultiFrameRecoveryRefiner
    if kind=='multimodal_event':
        from multimodal_event_refine import MultimodalEventRefiner
        return MultimodalEventRefiner
    if kind=='hybrid_motion':
        from hybrid_motion_refine import HybridMotionRefiner
        return HybridMotionRefiner
    if kind=='visual_edge':
        from visual_edge_refine import VisualEdgeRefiner
        return VisualEdgeRefiner
    if kind=='visual_motion':
        from visual_motion_refine import VisualMotionRefiner
        return VisualMotionRefiner
    if kind in ['inverse_dense','bidirectional_dense']:
        from inverse_dense_refine import InverseDenseRefiner,BidirectionalDenseRefiner
        return InverseDenseRefiner if kind=='inverse_dense' else BidirectionalDenseRefiner
    if kind=='visual_recovery':
        from visual_recovery import VisualVerifiedRecoveryRefiner
        return VisualVerifiedRecoveryRefiner
    if kind=='fast_recovery':
        from fast_recovery import FastVerifiedRecoveryRefiner
        return FastVerifiedRecoveryRefiner
    if kind=='joint':
        from joint_refine import JointRefiner
        return JointRefiner
    if kind=='image_event':
        from image_event_refine import ImageEventRefiner
        return ImageEventRefiner
    if kind=='transfer':
        from transfer_refine import TransferRefiner
        return TransferRefiner
    if kind=='trajectory':
        from trajectory_refine import TrajectoryRefiner
        return TrajectoryRefiner
    if kind=='recovery':
        from recover_joint import RecoveryRefiner
        return RecoveryRefiner
    if kind=='verified_recovery':
        from verified_recovery import VerifiedRecoveryRefiner
        return VerifiedRecoveryRefiner
    if kind=='dense_motion':
        from dense_motion_refine import DenseMotionRefiner
        return DenseMotionRefiner
    if kind=='motion':
        from motion_refine import MotionRefiner
        return MotionRefiner
    if kind=='retiming':
        from retime_divisions import RetimingRefiner
        return RetimingRefiner
    raise ValueError('Unknown stage: '+str(kind))

class LineagePipeline:
    def __init__(self,artifact_root,selection,data_root,cache_dir):
        self.root=Path(artifact_root);self.selection=selection;self.cache=Path(cache_dir);self.cache.mkdir(parents=True,exist_ok=True)
        torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
        base=selection['base'];self.base=EventRefiner([self.root/n for n in base['model_files']],data_root,base['config'],self.cache/'event_cache')
        self.stages=[]
        for i,stage in enumerate(selection['stages']):
            cls=refiner_class(stage['kind']);obj=cls([self.root/n for n in stage['model_files']],data_root,stage['config'],self.cache/'joint_cache')
            self.stages.append(obj)
        names=list(dict.fromkeys(n for stage in [base,*selection['stages']] for n in stage['model_files']))
        self.paths=[self.root/n for n in names];self.identity={'selection':selection,'model_sha256':{n:hashlib.sha256((self.root/n).read_bytes()).hexdigest() for n in names}}
    def refine(self,name,nodes,edges):
        start=time.time();nodes,edges,base_stats=self.base.refine(name,nodes,edges);stats={'base':base_stats,'stages':[]}
        for description,stage in zip(self.selection['stages'],self.stages):
            nodes,edges,report=stage.refine(name,nodes,edges);stats['stages'].append({'kind':description['kind'],**report})
        pairs=[(int(e['source_id']),int(e['target_id'])) for e in edges]
        assert len(pairs)==len(set(pairs)),'Duplicate edges'
        assert max(Counter(d for s,d in pairs).values(),default=0)<=1,'Invalid merge'
        assert max(Counter(s for s,d in pairs).values(),default=0)<=2,'Invalid division'
        assert all(s in nodes and d in nodes and nodes[d]['t']==nodes[s]['t']+1 for s,d in pairs),'Dangling or nonconsecutive edges'
        stats['pipeline_seconds']=time.time()-start;return nodes,edges,stats
