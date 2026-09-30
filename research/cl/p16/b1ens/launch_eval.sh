#!/bin/bash
# wait for the 11 LOEO/half trainings to finish, then launch all end-to-end runs concurrently
M=/workspace/cl/p16/b1ens/models; D=/workspace/cl/p16/b1ens
LOGS="$(ls $M/loeo_tr*_s[1-4].log) $(ls $M/half6bba_s*.log)"
while true; do n=0; for f in $LOGS; do grep -q TRAIN_DONE $f && n=$((n+1)); done; [ $n -eq 11 ] && break; sleep 20; done
echo "all 11 trainings done $(date -u)"
cd $D
FULLW="hold36:2,prev4:1,audit32:2,t127a:4,t127b:4"; HALFW="hold36:1,prev4:1,audit32:1,t127a:1,t127b:1"
for k in 1 2 3 4; do (setsid nohup ./b1e_driver.sh s$k "$M/loeo_tr{E}_s$k/last.pt" "$FULLW" > be_s$k.log 2>&1 < /dev/null &); done
(setsid nohup ./b1e_driver.sh E5 "$M/loeo_tr{E}_s*/last.pt" "$FULLW" > be_E5.log 2>&1 < /dev/null &)
for k in 0 1 2; do (setsid nohup ./b1e_driver.sh h$k "$M/half6bba_s$k/last.pt" "$HALFW" 44 > be_h$k.log 2>&1 < /dev/null &); done
echo "launched eval $(date -u)"
