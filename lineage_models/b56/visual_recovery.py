"""High-recall parent screening reuses the base pipeline's visual embeddings."""
from pathlib import Path
import hashlib,json
import numpy as np,torch,lightgbm as lgb
from verified_recovery import VerifiedRecoveryRefiner
from refine_events import EventRefiner,structure

class VisualVerifiedRecoveryRefiner(VerifiedRecoveryRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        assert len(paths)==5
        super().__init__(paths[:2],root,config,cache_dir);self.paths=[Path(p) for p in paths]
        checkpoint=torch.load(paths[2],map_location='cpu',weights_only=False);assert checkpoint['config']['architecture']=='parent_visual_prefilter_lgbm'
        self.gate=lgb.Booster(model_str=checkpoint['model_string']);key=str(self.config.get('gate_recall',.995));self.gate_threshold=checkpoint['thresholds'][key]
        self.parent_encoder=EventRefiner(paths[3:],root,cache_dir=self.cache.parent/'event_cache')
        salt=hashlib.sha256(self.paths[2].read_bytes()+str(self.gate_threshold).encode()).digest()
        self.recovery_signature=hashlib.sha256(self.paths[0].read_bytes()+salt).hexdigest()[:16]
        self.signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)+salt).hexdigest()[:16]
        for model in self.models:model.fast_photo_pool=True
    def proposals(self,name,nodes,edges):
        self.movie_name=name
        return super().proposals(name,nodes,edges)
    def filter_wanted(self,nodes,edges,wanted):
        ids,emb=self.parent_encoder.embeddings(self.movie_name,nodes);lookup={n:i for i,n in enumerate(ids)};out,_,_,_=structure(nodes,edges)
        keys=[s for s in wanted if out.get(s)];keep={s:ds for s,ds in wanted.items() if not out.get(s)}
        if keys:
            prob=self.gate.predict(emb[0][[lookup[s] for s in keys]],num_threads=2)
            keep.update({s:wanted[s] for s,p in zip(keys,prob) if p>=self.gate_threshold})
        print('VISUAL_PARENT_PREFILTER',self.movie_name,len(wanted),'->',len(keep),'threshold',self.gate_threshold,flush=True)
        return keep
