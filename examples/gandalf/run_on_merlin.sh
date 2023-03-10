#!/bin/bash
basedir=`pwd -P`
echo 'Curdir: '$basedir
# curfile=$0
# echo 'CurFile: '$curfile
grandparentdir="$(cd ../../; pwd)"
echo 'grandparentdir: '$grandparentdir
echo 'GPU NUM: '$ARNOLD_WORKER_GPU

export PYTHONPATH=$PYTHONPATH:$grandparent_dir
export PYTHONPATH=$PYTHONPATH:/opt/tiger
export PYTHONPATH=$PYTHONPATH:/opt/tiger/anyon2
export PYTHONPATH=$PYTHONPATH:/opt/tiger/titan
export PYTHONPATH=$PYTHONPATH:/opt/tiger/cruise
export PYTHONPATH=$PYTHONPATH:/opt/tiger/easyguard

# # pretrain weights
echo 'pulling down pretrained weights...'
if [ ! -d "models/weights" ]; then mkdir models/weights; fi
if [ ! -d "models/weights/fashion_deberta_asr" ]; then hadoop fs -get  hdfs:///user/jiangxubin/models/pretrain/fashion_deberta_asr ./models/weights; fi

# # trigger train proc
export CRUISE_HOME=${CRUISE_HOME:=/opt/tiger/cruise}
export GANDALF_HOME=${GANDALF_HOME:=$basedir}

echo 'CRUISE_HOME: '$CRUISE_HOME
if [ -f "$CRUISE_HOME/cruise/tools/TORCHRUN" ]; then
  $CRUISE_HOME/cruise/tools/TORCHRUN  $GANDALF_HOME/main.py $@
else
  echo 'CRUISE_HOME not exist'
fi