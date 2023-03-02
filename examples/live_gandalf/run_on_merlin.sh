#!/bin/bash
basedir=`cd $(dirname $0); pwd -P`
echo 'Main Path: '$basedir
echo 'GPU NUM: '$ARNOLD_WORKER_GPU

export PYTHONPATH=$PYTHONPATH:$basedir
export PYTHONPATH=$PYTHONPATH:/opt/tiger
export PYTHONPATH=$PYTHONPATH:/opt/tiger/anyon2
export PYTHONPATH=$PYTHONPATH:/opt/tiger/cruise
export PYTHONPATH=$PYTHONPATH:/opt/tiger/titan

# setup pretrained weights
cd $basedir || exit
# cd $basedir and show
pwd

# pretrain weights
echo 'pulling down pretrained weights...'
[ ! -d "models/weights" ] && mkdir models/weights
[ ! -d "models/weights/simcse_bert_base" ] && hadoop fs -get hdfs:///user/jiangxubin/models/pretrain/simcse_bert_base ./models/weights
[ ! -d "models/weights/fashion_deberta_asr" ] && hadoop fs -get  hdfs:///user/jiangxubin/models/pretrain/fashion_deberta_asr ./models/weights

# setup tensorboard
mkdir events_log
nohup tensorboard --logdir=./events_log --port=$ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all > tensorboard.log &

# trigger train proc
export TENSORBOARD_LOGDIR=""
export CRUISE_HOME=${CRUISE_HOME:=/opt/tiger/cruise}
export GANDALF_GUARD_HOME=${GANDALF_GUARD_HOME:=$basedir}
$CRUISE_HOME/cruise/tools/TORCHRUN main.py

# Trial运行结束后，实时tensorboard服务不再能查看，此时保存所有tensorboard log到hdfs，可在Task的Tensorboard接口查看
hadoop fs -rm -r $ARNOLD_OUTPUT/events_log
hadoop fs -put ./events_log $ARNOLD_OUTPUT

exit $ec