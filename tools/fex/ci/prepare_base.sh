set -x
echo "============= start prepare base ========="
bvc clone ss_bin -f /opt/tiger/ss_bin
bvc clone pyutil -f /opt/tiger/pyutil
bvc clone yarn_deploy -f /opt/tiger/yarn_deploy/
bvc clone consul_deploy -f /opt/tiger/consul_deploy/
bvc clone jdk -f /opt/tiger/jdk
bvc clone --version=1.0.0.42 lab/arnold/cuda -f /opt/tiger/ci_cuda
chown -R tiger:tiger /opt/tiger/yarn_deploy/
rm -rf ~/.bvc_store
rm -rf ~/.cache
pip install Jinja2 -i https://bytedpypi.byted.org/simple/ --no-cache-dir
runuser -l tiger -c 'export CONSUL_HTTP_HOST=10.27.80.81 && /opt/tiger/yarn_deploy/hadoop/bin/hdfs version' # 生成 hdfs 的配置，需要 tiger 账户

# fix curl
rm /usr/lib/x86_64-linux-gnu/libcurl.so.4
ln -s /usr/lib/x86_64-linux-gnu/libcurl.so.4.3.0 /usr/lib/x86_64-linux-gnu/libcurl.so.4
curl http://codebaseci.byted.org/ci/open/setup_consul.sh | bash

echo "============= prepare base complete  ========="
