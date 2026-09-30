#!/bin/bash
pkill -f "dn_driver.sh dnA"; pkill -f "dn_driver.sh dnB"; pkill -f "ps_dnA_"; pkill -f "ps_dnB_"; sleep 3
ps aux | grep -E "ps_dn[AB]_" | grep -v grep | wc -l
cd /workspace/cl/p16/divnet
python3 - <<'EOF'
p='/workspace/cl/p16/divnet/dn_driver.sh'; s=open(p).read()
old="s = s.replace(old, new).replace(\"sys.path.insert(0, str(Path(__file__).parent))\", \"sys.path.insert(0, '/workspace/p56stage')\")\n"
new=old+"s = s.replace(\"ngpu = max(1, min(2, torch.cuda.device_count()))\", \"ngpu = int(os.environ.get('DN_NW', '2'))\").replace(\"'CUDA_VISIBLE_DEVICES': str(g)\", \"'CUDA_VISIBLE_DEVICES': os.environ.get('DN_GPU', str(g))\")\n"
if 'DN_NW' not in s:
    assert s.count(old)==1; s=s.replace(old,new); open(p,'w').write(s)
EOF
grep -c "DN_NW" dn_driver.sh
mkdir -p /workspace/cl/p16/divnet/trash && mv ps_dnA_* ps_dnB_* ps_dnr_* /workspace/cl/p16/divnet/trash/ 2>/dev/null
export DN_B1LOEO="/workspace/cl/p16/divnet/b1loeo_tr{E}/last.pt" DN_GPU=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
(DN_NW=2 DN_SHORT=b1loeo:300 setsid nohup ./dn_driver.sh dnA divnet "/workspace/cl/p16/divnet/models/dn1_full_tr{E}_s*.pt" rank hold36,prev4,audit32,t127a,t127b > dn_dnA.log 2>&1 < /dev/null &)
sleep 3
(DN_NW=1 DN_SHORT=b1dep:300 setsid nohup ./dn_driver.sh dnB divnet "/workspace/cl/p16/divnet/models/dn1_full_tr{E}_s*.pt" rank hold36,prev4,audit32,t127a,t127b > dn_dnB.log 2>&1 < /dev/null &)
sleep 25; grep -c "ngpu = int" p_stage9dn.py; nvidia-smi --query-compute-apps=pid,used_memory --format=csv | wc -l
