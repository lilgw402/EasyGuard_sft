#!/usr/bin/env bash
set -x
# 参考自 data ci 的脚本
bvc_version=1.0.0.94
tiger_root=/opt/tiger
bvc_path=${tiger_root}/bvc
if ! type '$bvc' > /dev/null; then
    echo "========== install bvc ============"
    mkdir -p $bvc_path && cd $bvc_path
    wget -q -O bvc.tar.gz http://d.scm.byted.org/api/download/ceph:tao.modules.bvc_$bvc_version.tar.gz
    tar -xvzf bvc.tar.gz -C $bvc_path > /dev/null
    if [[ ${IDC} == 'boe' ]]; then
      # boe 使用单独的 cdn 否则在容器里面无法连接
      echo 'use boe cdn: [10.225.71.70, 10.225.71.82]'
      echo "cdn: [10.225.71.70, 10.225.71.82]" >> /opt/tiger/bvc/bin/config.yml
    fi
    echo "========== finish install bvc ============"
fi
export PATH="/opt/tiger/bvc/bin:${PATH}"

