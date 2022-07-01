#!/usr/bin/env bash

set -ex

if [[ "$ARNOLD_ROLE" != "worker" ]]; then
    # launcher / scheduler
    sleep infinity
fi

fex_root=$(dirname $(realpath $0))/../../../
cd $fex_root

min_nodes=${MIN_NODES:=1}
max_nodes=$ARNOLD_WORKER_NUM
rdzv_conf=${RDZV_CONF:="timeout=86400"}
max_restarts=${MAX_RESTARTS:=100000}
dataset_type=${DATASET_TYPE:="dali"}
max_app_attempts=${MAX_APP_ATTEMPTS:=3}

if [[ -z $CONFIG_PATH ]]; then
    echo "must specify env CONFIG_PATH"
    exit 1
fi
config_path=$CONFIG_PATH
output_path=${OUTPUT_PATH:="output"}

if [[ $FEX_USE_PTX_TSFM -eq 1 ]]; then
    # fast transformer
    export LD_LIBRARY_PATH=/opt/tiger/ptx__/ptx/ops/_lib/fastertransformer_training_v1/cu111/torch181:$LD_LIBRARY_PATH
fi

function start_train() {
    etcd_host=$ARNOLD_WORKER_0_HOST
    etcd_port=$ARNOLD_WORKER_0_PORT
    if [[ $ARNOLD_ID -eq 0 ]]; then
        # TODO: move to image
        etcd_binary=/opt/tiger/etcd/etcd
        if [[ ! -f $etcd_binary ]]; then
            echo "installing etcd"
            ETCD_VER=v3.5.2
            curl -L https://github.com/etcd-io/etcd/releases/download/${ETCD_VER}/etcd-${ETCD_VER}-linux-amd64.tar.gz -o /tmp/etcd-${ETCD_VER}-linux-amd64.tar.gz
            rm -rf /opt/tiger/etcd/ && mkdir -p /opt/tiger/etcd/
            tar xzvf /tmp/etcd-${ETCD_VER}-linux-amd64.tar.gz -C /opt/tiger/etcd/ --strip-components=1
            rm -f /tmp/etcd-${ETCD_VER}-linux-amd64.tar.gz
            $etcd_binary --version
        fi

        echo "node 0: starting etcd server"
        etcd_data_dir=/opt/tiger/etcd/data
        rm -rf $etcd_data_dir
        $etcd_binary --enable-v2 \
            --data-dir $etcd_data_dir \
            --listen-client-urls http://0.0.0.0:$etcd_port \
            --advertise-client-urls http://0.0.0.0:$etcd_port \
            --initial-cluster-state new \
            >/opt/tiger/etcd/server.log 2>&1 &
        etcd_pid=$!
    fi

    # wait for etcd
    sleep 60

    # remove proxy
    unset http_proxy https_proxy no_proxy

    # start watchdog
    python3 example/clip/elastic/watchdog.py 2>&1 &
    watchdog_pid=$!

    # run torch elastic
    set +e
    python3 -m torch.distributed.run \
        --nnodes=$min_nodes:$max_nodes \
        --nproc_per_node=$ARNOLD_WORKER_GPU \
        --rdzv_id=fex-job \
        --rdzv_backend=etcd \
        --rdzv_endpoint=$etcd_host:$etcd_port \
        --rdzv_conf=$rdzv_conf \
        --max_restarts=$max_restarts \
        example/clip/elastic/train.py \
        --config_path $config_path \
        --output_path $output_path \
        --dataset_type $dataset_type
    train_ret=$?
    set -e
}

function cleanup() {
    if [[ -n $etcd_pid ]]; then
        echo "killing etcd server (pid ${etcd_pid})"
        kill -9 $etcd_pid
        etcd_pid=
    fi
    if [[ -n $watchdog_pid ]]; then
        echo "killing watchdog (pid $watchdog_pid)"
        kill -9 $watchdog_pid
        watchdog_pid=
    fi
}

for attempt in $(seq $max_app_attempts); do
    echo "Start training (attempt $attempt)"
    start_train
    cleanup

    if [[ $train_ret -eq 0 ]]; then
        echo "Training process succeeded!"
        exit 0
    fi

    echo "Training process failed in attempt $attempt"
    sleep 60
done

exit 1
