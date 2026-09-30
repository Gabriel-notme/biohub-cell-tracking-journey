"""Fuse independently trained ranking and complete-candidate attention experts."""
from pathlib import Path
import hashlib,torch
from track_set_refine import TrackSetRefiner
from track_rank_refine import RankingHead
from track_set import TrackSetNet
from track_video import load_track
from joint_refine import JOINT_DEFAULT

class TrackEnsembleRefiner(TrackSetRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        torch.backends.mha.set_fastpath_enabled(False);self.paths=[Path(p) for p in paths];self.root=Path(root);self.cache=Path(cache_dir or '/kaggle/working/set_cache');self.cache.mkdir(parents=True,exist_ok=True);self.heads=[];expected=None;count=0
        for path in self.paths:
            ck=torch.load(path,map_location='cpu',weights_only=False);architecture=ck['config'].get('architecture')
            if architecture not in ['candidate_rank','candidate_set_attention']:break
            count+=1
            if expected is None:expected=ck['encoder_sha256_list']
            else:assert expected==ck['encoder_sha256_list']
            if architecture=='candidate_rank':self.heads.extend(RankingHead(m,t,ck['config']['input_dim']) for m,t in zip(ck['models'],ck['temperatures']))
            else:
                model=TrackSetNet(**ck['config']).cuda().eval();model.load_state_dict(ck['model']);self.heads.append(model)
        assert count>=2 and self.heads
        encoders=self.paths[count:];assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in encoders]==expected
        self.models=[load_track(p)[0] for p in encoders];self.encoder_signatures=[x[:16] for x in expected]
        self.signature=hashlib.sha256(b'candidate-ensemble-v1'+b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16];self.config={**JOINT_DEFAULT,**(config or {})}
