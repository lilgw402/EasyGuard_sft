FROM hub.byted.org/arnold/search.fex.base:81f85f774353ce8871a0e6e987351f65

# matx
RUN pip3 install https://d.scm.byted.org/api/v2/download/byted.matx.matxscript_1.5.0.5.tar.gz --no-cache-dir -i https://bytedpypi.byted.org/simple/
RUN pip3 install https://d.scm.byted.org/api/v2/download/byted.matx.matx_text_pip_wheels_1.5.0.1.tar.gz --no-cache-dir -i https://bytedpypi.byted.org/simple/
RUN pip3 install https://d.scm.byted.org/api/v2/download/data.mlsys.byted_vision_1.5.0.4.tar.gz --no-cache-dir -i https://bytedpypi.byted.org/simple/
RUN pip3 install https://d.scm.byted.org/api/v2/download/byted.matx.matx_pytorch_pip_wheels_1.5.0.5.tar.gz --no-cache-dir -i https://bytedpypi.byted.org/simple/

# tensorflow
RUN pip3 install byted-tensorflow==2.4.1 -i https://bytedpypi.byted.org/simple/

# train helper
RUN mkdir -p /opt/tiger/fex_helper
COPY docker/FEXRUN /opt/tiger/fex_helper
RUN chmod 755 /opt/tiger/fex_helper/FEXRUN

# bvc
COPY docker/setup_env.sh .
RUN bash ./setup_env.sh

# ENV
ENV PYTHONPATH="${PYTHONPATH}:/opt/tiger/fex"
ENV PATH="/opt/tiger/bvc/bin:/opt/tiger/fex_helper:${PATH}"
ENV CONSUL_HTTP_HOST=10.27.80.81
ENV https_proxy=http://bj-rd-proxy.byted.org:3128
ENV http_proxy=http://bj-rd-proxy.byted.org:3128
ENV SEC_KV_AUTH=1
