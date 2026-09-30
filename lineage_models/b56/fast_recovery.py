from pathlib import Path
import hashlib
import numpy as np,torch,lightgbm as lgb
from verified_recovery import VerifiedRecoveryRefiner
from refine_events import structure
from cell_event import chain,fork_geometry

class FastVerifiedRecoveryRefiner(VerifiedRecoveryRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        assert len(paths)==3
        super().__init__(paths[:2],root,config,cache_dir);self.paths=[Path(p) for p in paths]
        for model in self.models:model.fast_photo_pool=True
        checkpoint=torch.load(paths[2],map_location='cpu',weights_only=False);assert checkpoint['config']['architecture']=='geometry_prefilter_lgbm'
        self.geometry_gate=lgb.Booster(model_str=checkpoint['model_string']);self.gate_threshold=checkpoint['threshold']
        gate_hash=hashlib.sha256(self.paths[2].read_bytes()).digest()
        self.recovery_signature=hashlib.sha256(self.paths[0].read_bytes()+gate_hash).hexdigest()[:16]
        self.signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
    def filter_wanted(self,nodes,edges,wanted):
        out,prev,_,pos=structure(nodes,edges);rows=[];keys=[];keep={s:list(ds) for s,ds in wanted.items() if not out.get(s)}
        for s,ds in wanted.items():
            old=out.get(s,[])
            if not old:continue
            h=chain(s,prev,pos);first=chain(old[0],out,pos)
            for d,expected in ds:
                second=np.vstack([expected,chain(d,out,pos)[:3]]);rows.append(fork_geometry(h,first,second));keys.append((s,d,expected))
        if rows:
            probabilities=self.geometry_gate.predict(np.asarray(rows,np.float32),num_threads=2)
            for (s,d,expected),p in zip(keys,probabilities):
                if p>=self.gate_threshold:keep.setdefault(s,[]).append((d,expected))
        print('GEOMETRY_PREFILTER',len(wanted),'sources ->',len(keep),'threshold',self.gate_threshold,flush=True)
        return keep
