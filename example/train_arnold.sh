#!/usr/bin/env bash

set -e

# ======== 一些环境变量，主要和 torch 在 arnold 上分布式训练使用 NCCL 有关 ====
export SEC_KV_AUTH=1 # 和abase 权限有关
export LIBHDFS_OPTS="-Dhadoop.root.logger=${HADOOP_ROOT_LOGGER:-ERROR,console}" # 和hdfs log 有关
export NCCL_IB_GID_INDEX=3 # 和 pytorch on arnold 训练有关
export NCCL_IB_HCA="mlx5_2"
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="eth0"
export BYTED_TORCH_FX="O0" # 涉及到 bytetorch的一个优化，之前测试有些情况有问题，先关掉。

# ======== 对 FEX_USE_PTX_TSFM 和 ptx 环境的处理，用来做 faster transformer kernel 训练加速 =====
if [ -z $FEX_USE_PTX_TSFM ]; then
    FEX_USE_PTX_TSFM=0
fi
if [ $FEX_USE_PTX_TSFM -eq 1 ];
    then
        echo "FEX_USE_PTX_TSFM=1, will use ptx transformer kernel to boost training speed"
        # 确保CUDA_HOME
        if [ -d "/usr/local/cuda" ]; then
            export CUDA_HOME="/usr/local/cuda"
        fi
        if [ -L "/usr/local/cuda" ] && [ -d "/opt/tiger/cuda" ] && [ "${ARNOLD_TRIAL_ID}" != "" ]; then
            sudo rm "/usr/local/cuda"
            sudo ln -s "/opt/tiger/cuda" "/usr/local/cuda"
        fi

        # 安装PTX
        export PYTHONPATH=/opt/tiger/ptx__:$PYTHONPATH
        export LD_LIBRARY_PATH="$(python3 -c 'from ptx.ops.fastertransformer.training import FT_TRAIN_LIB_PATH; print(FT_TRAIN_LIB_PATH)'):$LD_LIBRARY_PATH"
fi    

# ======= 对输入做一些 echo =============
train_script=${1:-"./example/clip/train.py"}

echo "train_script [$train_script]"
echo "argument [$@]"
i=1;
for agm in "$@" 
do
  echo "argument - $i: [$agm]";
  i=$((i + 1));
done

cd /opt/tiger/fex # 默认放到 /opt/tiger 这了
pip3 install --editable . --index-url=https://bytedpypi.byted.org/simple/  # 先安装一下

# ============ 训练启动 ===================
echo "start launch tasks ..."
FEXRUN $@ || exit 1
echo "finish executing ..."

sleep 10
