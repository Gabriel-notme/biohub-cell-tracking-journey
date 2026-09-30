#!/bin/bash
# usage: variant_pp.sh NAME SRC_RUN MOVIES NB [ILPDIR] [NPROC]
set -x
NAME=$1; SRC=$2; MOVIES=$3; NB=$4; ILPDIR=$5; NP=$6
[ -z "$NP" ] && NP=8
R=/workspace/runs/$NAME
mkdir -p $R/working
rsync -a --exclude lineage_graphs --exclude 'lineage_cache_*' --exclude lineage_claims --exclude reference_graphs --exclude submission.csv --exclude 'lineage_stream*' --exclude 'lineage_runtime_validation.json' $SRC/working/ $R/working/
rm -f $R/working/reference_postprocessing_complete.json $R/working/biohub_live_resume/base_submission_state.json $R/working/biohub_live_resume/final_submission_state.json $R/working/biohub_live_resume/base_submission.log
if [ -n "$ILPDIR" ] && [ "$ILPDIR" != "none" ]; then
  D=$R/working/tracking_repo/predictions/unknown/unet_transformer/split_0
  for g in $ILPDIR/*.geff; do rm -rf $D/$(basename $g); cp -r $g $D/; done
  echo ILP_REPLACED $(ls $ILPDIR | wc -l)
fi
cd /workspace/code
python run_nb2.py $NB $MOVIES $R > /workspace/logs/$NAME.nb.log 2>&1
echo NB_RC $?
ls $R/working/reference_graphs | wc -l
LIN_RUNW=$R/working python lin_run.py $NAME $MOVIES $R/lin $NP none > /workspace/logs/$NAME.lin.log 2>&1
echo LIN_RC $?
grep -E "^$NAME " /workspace/logs/$NAME.lin.log
echo VARIANT_DONE
