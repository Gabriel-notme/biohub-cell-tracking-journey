"""Detector-aligned supervised events on fitting movies only, cloud execution."""
from mine_events import *
import shutil
SOURCE=DATA;DEST=ROOT/'events_aligned';DEST.mkdir(exist_ok=True)
def run_alignment():
    torch.set_num_threads(3);torch.backends.cudnn.benchmark=True
    split=json.loads((SOURCE/'split.json').read_text());shutil.copy2(SOURCE/'split.json',DEST/'split.json')
    for name in split['calibration']+split['audit']:
        for suffix in ['.npz','_patches.npy']:
            dst=DEST/(name+suffix)
            if not dst.exists():dst.symlink_to(SOURCE/(name+suffix))
    model,_,_=load_model(REPO/'weights/unet_transformer/split_0/edge_predictor_best.pth',torch.device('cuda'));model.eval()
    reports=[];started=time.time();detroot=ROOT/'training_detector_candidates';detroot.mkdir(exist_ok=True)
    for name in split['train']:
        marker=DEST/(name+'_aligned.json')
        if marker.exists():reports.append(json.loads(marker.read_text()));continue
        nodes,out,prev,frames,pos=graph(TRAIN/(name+'.geff'));movie=Movie(TRAIN/(name+'.zarr'))
        dp=detroot/(name+'.npz')
        if dp.exists():
            with np.load(dp) as d:det={int(k):d[k] for k in d.files}
        else:
            det=detections(model,movie);np.savez_compressed(dp,**{str(k):v for k,v in det.items()})
        trees={t:cKDTree(c*SCALE) for t,c in det.items() if len(c)}
        matched={};assigned={};alignedpos=dict(pos)
        # Only known labeled cells become positive examples. Unlabeled cells
        # are not used as positive annotations or assumed to be background.
        for t,ns in frames.items():
            if t not in trees:continue
            proposals=[]
            for n in ns:
                distance,j=trees[t].query(pos[n])
                if distance<=4.5:proposals.append((distance,n,int(j)))
            used=set()
            for distance,n,j in sorted(proposals):
                if j in used:continue
                used.add(j);matched[n]=(t,j);assigned[(t,j)]=n;alignedpos[n]=det[t][j]*SCALE
        with np.load(SOURCE/(name+'.npz')) as data:old={k:data[k] for k in data.files}
        oldids=old['node_ids'];oldcount=len(oldids);coords=[];keys={};phase=[];newids=[]
        def index(t,j):
            key=(int(t),int(j))
            if key not in keys:
                keys[key]=oldcount+len(coords);coords.append((t,det[t][j]))
                gt=assigned.get(key);phase.append(float(len(out.get(gt,[]))==2) if gt is not None else -1.)
                newids.append(int(oldids.max())+1000000+len(coords))
            return keys[key]
        def future(t,j):
            points=[det[t][j]*SCALE]
            for dt in range(1,4):
                if t+dt not in trees:break
                distance,k=trees[t+dt].query(points[-1])
                if distance>7.5:break
                points.append(det[t+dt][k]*SCALE)
            return np.asarray(points,np.float32)
        rng=np.random.default_rng(20260923);er=[];eg=[];fr=[];fg=[]
        sources=[s for s,children in out.items() if s in matched and any(c in matched for c in children)]
        if len(sources)>250:sources=rng.choice(sources,250,replace=False).tolist()
        sources=sorted(set(sources)|{s for s,c in out.items() if len(c)==2 and s in matched})
        for s in sources:
            t=int(nodes[s]['t']);children=[c for c in out[s] if nodes[c]['t']==t+1];valid=[c for c in children if c in matched]
            if not valid:continue
            h=chain(s,prev,alignedpos);si=index(*matched[s])
            for c in valid:
                er.append((si,index(*matched[c]),1));eg.append(edge_geometry(h,chain(c,out,alignedpos)))
            if len(children)==2 and len(valid)==2:
                a,b=valid;fr.append((si,index(*matched[a]),index(*matched[b]),1))
                fg.append(fork_geometry(h,chain(a,out,alignedpos),chain(b,out,alignedpos)))
            if t+1 not in trees:continue
            near=trees[t+1].query_ball_point(alignedpos[s],11.)
            near=[j for j in near if min(np.linalg.norm(det[t+1][j]*SCALE-pos[c]) for c in children)>4.5]
            near.sort(key=lambda j:np.linalg.norm(det[t+1][j]*SCALE-alignedpos[s]))
            for j in near[:2]:
                ff=future(t+1,j);di=index(t+1,j);er.append((si,di,0));eg.append(edge_geometry(h,ff))
                if len(ff)>=2:
                    a=valid[0];fr.append((si,index(*matched[a]),di,0));fg.append(fork_geometry(h,chain(a,out,alignedpos),ff))
        sourcebank=np.load(SOURCE/(name+'_patches.npy'),mmap_mode='r')
        bank=np.lib.format.open_memmap(DEST/(name+'_patches.npy'),mode='w+',dtype=np.float16,shape=(oldcount+len(coords),5,12,24,24))
        bank[:oldcount]=sourcebank
        for i in sorted(range(len(coords)),key=lambda i:coords[i][0]):
            t,c=coords[i];bank[oldcount+i]=movie.patch(t,c)
        bank.flush();del bank,sourcebank
        newedge=np.asarray(er,np.int32).reshape(-1,3);newfork=np.asarray(fr,np.int32).reshape(-1,4)
        np.savez_compressed(DEST/(name+'.npz'),edges=np.r_[old['edges'],newedge],edge_geom=np.r_[old['edge_geom'],np.asarray(eg,np.float32).reshape(-1,16)],
            forks=np.r_[old['forks'],newfork],fork_geom=np.r_[old['fork_geom'],np.asarray(fg,np.float32).reshape(-1,28)],
            phase=np.r_[old['phase'],np.asarray(phase,np.float32)],node_ids=np.r_[oldids,newids],split='train')
        record={'movie':name,'added_cells':len(coords),'added_edges':len(er),'added_edge_positive':int(newedge[:,-1].sum()),
            'added_forks':len(fr),'added_fork_positive':int(newfork[:,-1].sum()),'seconds':round(time.time()-started)}
        marker.write_text(json.dumps(record));reports.append(record);print('ALIGNED',json.dumps(record),flush=True)
        (DEST/'alignment_report.json').write_text(json.dumps(reports,indent=2))
    (DEST/'alignment_complete.json').write_text(json.dumps({'movies':len(reports),'seconds':time.time()-started}))
if __name__=='__main__':run_alignment()
