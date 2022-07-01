#!/usr/bin/env bash

set -ex

fex_root=$(dirname $(realpath $0))/../
cd $fex_root

if [[ -z $ARNOLD_WORKER_GPU ]]; then
    echo "ARNOLD_WORKER_GPU not set. Please submit elastic training jobs on arnold."
    exit 1
fi

if [[ "$ARNOLD_ROLE" != "worker" ]]; then
    # launcher / scheduler
    sleep infinity
fi

if [[ $ARNOLD_WORKER_NUM -eq 1 && $ARNOLD_WORKER_GPU -eq 1 ]]; then
    # run local
    python3 $@
    exit 0
fi

min_nodes=${MIN_NODES:-1}
max_nodes=$ARNOLD_WORKER_NUM
rdzv_conf=${RDZV_CONF:-"timeout=86400"}
max_restarts=${MAX_RESTARTS:-100000}

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
python3 -m torch.distributed.run \
    --nnodes=$min_nodes:$max_nodes \
    --nproc_per_node=$ARNOLD_WORKER_GPU \
    --rdzv_id=fex-job \
    --rdzv_backend=etcd \
    --rdzv_endpoint=$etcd_host:$etcd_port \
    --rdzv_conf=$rdzv_conf \
    --max_restarts=$max_restarts \
    $@

# clean up
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
