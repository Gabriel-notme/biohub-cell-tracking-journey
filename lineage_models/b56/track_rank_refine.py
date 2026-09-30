from pathlib import Path
import hashlib,torch,numpy as np,lightgbm as lgb
from track_set_refine import TrackSetRefiner
from track_video import load_track
from joint_refine import JOINT_DEFAULT

class RankingHead:
    def __init__(self,booster,temperature,input_dim):self.booster=lgb.Booster(model_str=booster);self.temperature=temperature;self.input_dim=input_dim
    def __call__(self,x,valid):
        b,k,d=x.shape;flat=x.float().cpu().numpy().reshape(-1,d);score=torch.as_tensor(self.booster.predict(flat,num_threads=2).reshape(b,k),device=x.device,dtype=torch.float32)/self.temperature;score=score.masked_fill(~valid,-1e4);prob=score.softmax(1).clamp(1e-6,1-1e-6)
        edge=prob.log()-torch.log1p(-prob);count=x.new_tensor([-10.,10.,-10.])[None].expand(b,3).float();pair=torch.zeros((b,k,k),device=x.device,dtype=torch.float32)
        return edge,count,pair

class TrackRankRefiner(TrackSetRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        torch.backends.mha.set_fastpath_enabled(False);self.paths=[Path(p) for p in paths];self.root=Path(root);self.cache=Path(cache_dir or '/kaggle/working/set_cache');self.cache.mkdir(parents=True,exist_ok=True);ck=torch.load(paths[0],map_location='cpu',weights_only=False);assert ck['config']['architecture']=='candidate_rank'
        self.heads=[RankingHead(m,t,ck['config']['input_dim']) for m,t in zip(ck['models'],ck['temperatures'])];expected=ck['encoder_sha256_list'];assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in self.paths[1:]]==expected
        self.models=[load_track(p)[0] for p in self.paths[1:]];self.encoder_signatures=[x[:16] for x in expected];self.signature=hashlib.sha256(b'candidate-rank-v1'+b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16];self.config={**JOINT_DEFAULT,**(config or {})}
