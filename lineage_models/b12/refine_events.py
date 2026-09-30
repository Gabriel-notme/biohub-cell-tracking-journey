"""Image-conditioned association and division refinement with no node-count manipulation."""
from pathlib import Path
from collections import Counter
import hashlib, json, os, time
from itertools import combinations
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from cell_event import SCALE,Movie,chain,edge_geometry,fork_geometry,load_event_model

DEFAULT_CONFIG={'edge_weight':1.,'change_penalty':.4,'edge_min_prob':.65,'max_distance':10.,
                'fork_threshold':.80,'fork_min_branch':2,'fork_max_distance':13.,
                'fork_base_bonus':0.,'ensemble':'mean','temperature':1.,'fork_reassign':False,
                'fork_from_endpoints':False,
                'fork_veto_threshold':0.,
                'fork_reassign_old_max':.30,'fork_reassign_new_min':.70}

def structure(nodes,edges):
    out={};prev={};frames={}
    for e in edges:
        s,d=int(e['source_id']),int(e['target_id']);out.setdefault(s,[]).append(d);prev[d]=s
    for n,v in nodes.items():frames.setdefault(int(v['t']),[]).append(n)
    pos={n:np.array([v['z'],v['y'],v['x']],np.float32)*SCALE for n,v in nodes.items()}
    return out,prev,frames,pos

def batch_edge_geometry(rows,prev,out,pos):
    if not rows:return np.empty((0,16),np.float32)
    sources={s for s,d in rows};targets={d for s,d in rows}
    histories={s:chain(s,prev,pos) for s in sources};futures={d:chain(d,out,pos) for d in targets}
    velocity={s:(h[0]-h[-1])/max(1,len(h)-1) for s,h in histories.items()}
    next_velocity={d:(f[-1]-f[0])/max(1,len(f)-1) for d,f in futures.items()}
    delta=np.stack([pos[d]-pos[s] for s,d in rows]);v=np.stack([velocity[s] for s,d in rows]);w=np.stack([next_velocity[d] for s,d in rows])
    return np.column_stack([delta/10,np.linalg.norm(delta,axis=1)/10,v/10,(delta-v)/10,w/10,
        np.linalg.norm(w-v,axis=1)/10,[len(histories[s])>1 for s,d in rows],[len(futures[d])>1 for s,d in rows]]).astype(np.float32)

class EventRefiner:
    def __init__(self,models,root,config=None,cache_dir=None):
        torch.set_num_threads(4)
        self.paths=[Path(p) for p in (models.split(',') if isinstance(models,str) else models)]
        self.models=[];self.root=Path(root);self.config={**DEFAULT_CONFIG,**(config or {})}
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        # Match Kaggle T4 deployment precision during cloud validation too.
        self.amp_dtype=torch.float16
        for path in self.paths:
            model,ckpt=load_event_model(path,self.device);self.models.append(model)
        self.cache_dir=Path(cache_dir or os.environ.get('BIOHUB_EVENT_CACHE','/kaggle/working/event_cache'))
        self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.last_cache=None
    @torch.inference_mode()
    def embeddings(self,name,nodes):
        ids=sorted(nodes,key=lambda n:(nodes[n]['t'],n))
        signature=hashlib.sha256(json.dumps([(n,nodes[n]['t'],nodes[n]['z'],nodes[n]['y'],nodes[n]['x']) for n in ids]).encode()).hexdigest()[:16]
        model_signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)+str(self.amp_dtype).encode()).hexdigest()[:16]
        path=self.cache_dir/(name+'_'+signature+'_'+model_signature+'.npz')
        if path.exists():
            d=np.load(path);return ids,[d['m'+str(i)] for i in range(len(self.models))]
        movie=Movie(self.root/(name+'.zarr'),context=max(5,max(m.context for m in self.models)));result=[[] for _ in self.models]
        for start in range(0,len(ids),192):
            chosen=ids[start:start+192]
            x=np.stack([movie.patch(nodes[n]['t'],[nodes[n]['z'],nodes[n]['y'],nodes[n]['x']]) for n in chosen])
            x=torch.from_numpy(x).to(self.device,dtype=torch.float32)
            for j,model in enumerate(self.models):
                with torch.autocast('cuda',dtype=self.amp_dtype,enabled=self.device=='cuda'):
                    emb=model.encode(x)
                result[j].append(emb.float().cpu().numpy())
        result=[np.concatenate(r) for r in result]
        temporary=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz')
        np.savez_compressed(temporary,**{'m'+str(i):r for i,r in enumerate(result)})
        os.replace(temporary,path)
        return ids,result
    @torch.inference_mode()
    def score(self,kind,rows,geometry,embeddings,lookup):
        if not rows:return np.empty(0,np.float32)
        indices=np.asarray([[lookup[n] for n in r] for r in rows],np.int64)
        geometry=np.asarray(geometry,np.float32);results=[]
        for model,bank in zip(self.models,embeddings):
            chunks=[]
            for start in range(0,len(rows),2048):
                ix=indices[start:start+2048];g=torch.from_numpy(geometry[start:start+2048]).to(self.device)
                z=[torch.from_numpy(bank[ix[:,i]]).to(self.device) for i in range(ix.shape[1])]
                logits=model.edge_logits(*z,g) if kind=='edge' else model.fork_logits(*z,g)
                chunks.append(logits.float().cpu().numpy())
            results.append(np.concatenate(chunks))
        stacked=np.stack(results)
        logits=stacked.mean(0)
        mode=self.config.get(kind+'_ensemble',self.config['ensemble'])
        if mode=='conservative':logits=.65*logits+.35*stacked.min(0)
        elif mode=='last':logits=stacked[-1]
        elif mode=='first':logits=stacked[0]
        elif mode=='max_fork' and kind=='fork':logits=stacked.max(0)
        return 1/(1+np.exp(-np.clip(logits/self.config['temperature'],-30,30)))
    def refine(self,name,nodes,edges):
        started=time.time();cfg=self.config;out,prev,frames,pos=structure(nodes,edges)
        ids,emb=self.embeddings(name,nodes);lookup={n:i for i,n in enumerate(ids)}
        original={(int(e['source_id']),int(e['target_id'])):e for e in edges}
        fork_sources={s for s,d in out.items() if len(d)>1}
        locked_targets={d for s in fork_sources for d in out[s]}
        proposals=[]
        edge_frames=sorted(frames.items()) if cfg['edge_weight']>0 or cfg['fork_reassign'] else []
        for t,src in edge_frames:
            targets=frames.get(t+1,[])
            if not targets:continue
            tree=cKDTree([pos[n] for n in targets])
            for s in src:
                if s in fork_sources:continue
                cand=tree.query_ball_point(pos[s],cfg['max_distance'])
                for j in cand:
                    d=targets[j]
                    if d in locked_targets:continue
                    proposals.append((s,d))
        geom=batch_edge_geometry(proposals,prev,out,pos)
        probs=self.score('edge',proposals,geom,emb,lookup)
        learned=dict(zip(proposals,probs));selected=[dict(e) for (s,d),e in original.items() if s in fork_sources]
        learned_targets={}
        for (s,d),prob in learned.items():
            if prob>=cfg['edge_min_prob']:learned_targets.setdefault(s,set()).add(d)
        changes=0
        if cfg['edge_weight']<=0:selected=[dict(e) for e in edges]
        for t,src0 in (sorted(frames.items()) if cfg['edge_weight']>0 else []):
            src=[s for s in src0 if s not in fork_sources]
            tgt=[d for d in frames.get(t+1,[]) if d not in locked_targets]
            if not src or not tgt:continue
            # Include a per-source disappearance option instead of forcing a bad link.
            cost=np.full((len(src),len(tgt)+len(src)),10000.,np.float64)
            target_index={d:j for j,d in enumerate(tgt)}
            for i,s in enumerate(src):
                h=chain(s,prev,pos);v=(h[0]-h[-1])/max(1,len(h)-1)
                possible=learned_targets.get(s,set())|set(out.get(s,[]))
                for d in possible:
                    if d not in target_index:continue
                    j=target_index[d]
                    old=(s,d) in original
                    prob=float(learned.get((s,d),0))
                    if not old and prob<cfg['edge_min_prob']:continue
                    raw=np.linalg.norm(pos[d]-pos[s])
                    if raw>cfg['max_distance'] and not old:continue
                    motion=np.linalg.norm(pos[d]-pos[s]-.5*v)
                    cost[i,j]=motion+.05*raw+cfg['edge_weight']*(-np.log(max(prob,1e-5)))+(0 if old else cfg['change_penalty'])
                cost[i,len(tgt)+i]=12.+cfg['edge_weight']*3
            r,c=linear_sum_assignment(cost)
            for i,j in zip(r,c):
                if j>=len(tgt) or cost[i,j]>=1000:continue
                s,d=src[i],tgt[j];p=float(learned.get((s,d),0))
                if (s,d) in original:e=dict(original[s,d])
                else:
                    e={'source_id':s,'target_id':d,'edge_prob':p,'event_edge':1};changes+=1
                selected.append(e)
        # Mitosis is a scored three-cell event, after association and before export.
        out2,prev2,frames,pos=structure(nodes,selected)
        candidates=[];fg=[];double_flags=[]
        for t,parents in sorted(frames.items()):
            targets=frames.get(t+1,[])
            # New branches must have consistent downstream support.
            starts=[d for d in targets if (d not in prev2 or (cfg['fork_reassign'] and len(out2.get(prev2[d],[]))==1
                    and learned.get((prev2[d],d),1.)<cfg['fork_reassign_old_max']))
                    and len(chain(d,out2,pos))>=cfg['fork_min_branch']]
            if not starts:continue
            tree=cKDTree([pos[d] for d in starts])
            for s in parents:
                children=out2.get(s,[])
                if not children and cfg['fork_from_endpoints'] and s in prev2:
                    near=tree.query_ball_point(pos[s],cfg['fork_max_distance'])
                    near=sorted((starts[j] for j in near if starts[j] not in prev2),key=lambda d:np.linalg.norm(pos[d]-pos[s]))[:6]
                    for a,b in combinations(near,2):
                        if np.linalg.norm(pos[a]-pos[b])>20:continue
                        candidates.append((s,a,b));double_flags.append(True)
                        fg.append(fork_geometry(chain(s,prev2,pos),chain(a,out2,pos),chain(b,out2,pos)))
                    continue
                if len(children)!=1:continue
                a=children[0];aa=chain(a,out2,pos)
                if len(aa)<cfg['fork_min_branch']:continue
                for j in tree.query_ball_point(pos[s],cfg['fork_max_distance']):
                    b=starts[j]
                    if b==a:continue
                    if b in prev2 and learned.get((s,b),0.)<cfg['fork_reassign_new_min']:continue
                    if np.linalg.norm(pos[a]-pos[b])>20:continue
                    candidates.append((s,a,b));double_flags.append(False);fg.append(fork_geometry(chain(s,prev2,pos),aa,chain(b,out2,pos)))
        fp=self.score('fork',candidates,fg,emb,lookup)
        taken_s=set();taken_d=set();added=0;reassigned=0
        for i in np.argsort(fp)[::-1]:
            if fp[i]<cfg['fork_threshold']:break
            s,a,b=candidates[i]
            if s in taken_s or b in taken_d or a in taken_d:continue
            if b in prev2:
                old=prev2[b]
                if old in taken_s:continue
                selected=[e for e in selected if not(int(e['source_id'])==old and int(e['target_id'])==b)]
                reassigned+=1
            taken_s.add(s);taken_d.add(b)
            if double_flags[i]:
                taken_d.add(a)
                selected.append({'source_id':s,'target_id':a,'edge_prob':float(fp[i]),'event_division':1})
            selected.append({'source_id':s,'target_id':b,'edge_prob':float(fp[i]),'event_division':1});added+=1
        # Audit inherited divisions as well as adding missing ones. A rejected
        # division keeps the better image-conditioned daughter association.
        removed=0
        if cfg['fork_veto_threshold']>0:
            triples=[(s,*out[s]) for s in fork_sources if len(out[s])==2]
            fgeom=[fork_geometry(chain(s,prev,pos),chain(a,out,pos),chain(b,out,pos)) for s,a,b in triples]
            fprob=self.score('fork',triples,fgeom,emb,lookup)
            rejected=[row for row,p in zip(triples,fprob) if p<cfg['fork_veto_threshold']]
            epairs=[(s,d) for s,a,b in rejected for d in [a,b]]
            egeom=[edge_geometry(chain(s,prev,pos),chain(d,out,pos)) for s,d in epairs]
            eprob=self.score('edge',epairs,egeom,emb,lookup)
            discard=set()
            for i,(s,a,b) in enumerate(rejected):
                discard.add((s,b if eprob[2*i]>=eprob[2*i+1] else a));removed+=1
            selected=[e for e in selected if (int(e['source_id']),int(e['target_id'])) not in discard]
        # Structural checks also apply on hidden test data.
        pairs={(int(e['source_id']),int(e['target_id'])) for e in selected}
        assert len(pairs)==len(selected)
        assert max(Counter(d for s,d in pairs).values(),default=0)<=1
        assert max(Counter(s for s,d in pairs).values(),default=0)<=2
        assert all(s in nodes and d in nodes and nodes[d]['t']==nodes[s]['t']+1 for s,d in pairs)
        stats={'event_edge_candidates':len(proposals),'event_edges_changed':changes,'event_fork_candidates':len(candidates),
               'event_divisions_added':added,'event_divisions_removed':removed,'event_divisions_reassigned':reassigned,'event_seconds':round(time.time()-started,2)}
        stats['edge_probability_quantiles']=np.percentile(probs,[0,25,50,75,95,100]).tolist() if len(probs) else []
        stats['fork_probability_quantiles']=np.percentile(fp,[0,25,50,75,95,100]).tolist() if len(fp) else []
        print('EVENT_REFINEMENT',name,json.dumps(stats),flush=True)
        return nodes,selected,stats
