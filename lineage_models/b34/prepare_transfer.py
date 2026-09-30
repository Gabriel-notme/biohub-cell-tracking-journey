from pathlib import Path
import os,json,time,argparse
os.environ.setdefault('OMP_NUM_THREADS','3');os.environ.setdefault('POLARS_MAX_THREADS','2')
import numpy as np,torch
from scipy.spatial import cKDTree
from prepare_events import graph
from cell_event import SCALE,Movie,load_event_model
R=Path('/workspace/biohub');TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train');SOURCE=R/'joint_aligned';DEST=R/'transfer_data'

@torch.inference_mode()
def prepare(name,model):
    marker=DEST/(name+'_ready.json')
    if marker.exists():return json.loads(marker.read_text())
    start=time.time();nodes,out,prev,frames,pos=graph(TRAIN/(name+'.geff'))
    with np.load(SOURCE/(name+'.npz')) as d:data={k:d[k] for k in d.files}
    with np.load(R/'joint_data'/(name+'.npz')) as d:original_count=len(d['source_ids'])
    with np.load(R/'training_detector_candidates'/(name+'.npz')) as d:det={int(k):d[k] for k in d.files}
    aligned=dict(pos)
    for t,ns in frames.items():
        if t not in det or not len(det[t]):continue
        tree=cKDTree(det[t]*SCALE);proposals=[]
        for n in ns:
            distance,j=tree.query(pos[n])
            if distance<=4.5:proposals.append((distance,n,int(j)))
        used=set()
        for distance,n,j in sorted(proposals):
            if j in used:continue
            used.add(j);aligned[n]=det[t][j]*SCALE
    requests=[];lookup={}
    def request(t,p):
        coord=p/SCALE;key=(int(t),*np.rint(coord*32).astype(int).tolist())
        if key not in lookup:lookup[key]=len(requests);requests.append((int(t),coord))
        return lookup[key]
    metadata={}
    for kind in ['edge','fork']:
        indices=[]
        for row,relative in zip(data[kind+'_rows'],data[kind+'_rel']):
            ix=int(row[0]);s=int(data['source_ids'][ix]);t=int(nodes[s]['t']);origin=(aligned if ix>=original_count else pos)[s]
            indices.append([request(t,origin)]+[request(t+1,origin+delta) for delta in relative[1:]])
        metadata[kind+'_rows']=np.asarray(indices,np.int64);metadata[kind+'_y']=data[kind+'_rows'][:,-1];metadata[kind+'_geom']=data[kind+'_geom']
    order=sorted(range(len(requests)),key=lambda i:requests[i][0]);movie=Movie(TRAIN/(name+'.zarr'),context=7);bank=np.lib.format.open_memmap(DEST/(name+'_features.npy'),mode='w+',dtype=np.float16,shape=(len(requests),3,64))
    for starti in range(0,len(order),192):
        ix=order[starti:starti+192];x=torch.from_numpy(np.stack([movie.patch(*requests[i]) for i in ix])).cuda().float()
        with torch.autocast('cuda',dtype=torch.float16):z=model.encode(torch.cat([x[:,0:3],x[:,2:5],x[:,4:7]],0))
        z=z.reshape(3,len(ix),64).transpose(0,1).float().cpu().numpy();bank[ix]=z
    bank.flush();del bank;np.savez_compressed(DEST/(name+'.npz'),**metadata)
    row={'movie':name,'image_queries':len(requests),'seconds':round(time.time()-start,1)};marker.write_text(json.dumps(row));return row

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2);a=p.parse_args();torch.set_num_threads(3);torch.backends.cudnn.benchmark=True
    DEST.mkdir(exist_ok=True);split=json.loads((SOURCE/'split.json').read_text());(DEST/'split.json').write_text(json.dumps(split,indent=2));model,_=load_event_model(R/'B1/best.pt');names=(split['calibration']+split['train'])[a.shard::a.shards]
    for n in names:print('TRANSFER_PREPARED',json.dumps(prepare(n,model)),flush=True)
    (DEST/('shard_'+str(a.shard)+'_ready.json')).write_text(json.dumps({'ready':True,'movies':len(names)}))
    if all((DEST/('shard_'+str(i)+'_ready.json')).exists() for i in range(a.shards)):(DEST/'ready.json').write_text(json.dumps({'ready':True,'shards':a.shards}))
