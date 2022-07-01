FROM hub.byted.org/arnold/search.fex:14d1f0d63758b267c90edc5b8250fe74

RUN \
    bvc clone nlp/lib/ptx2 /opt/tiger/ptx__ --version 1.0.0.69 && \
    pip install torch~=1.8.0 torchvision~=0.9.0 torchelastic==0.2.0

ENV PYTHONPATH $PYTHONPATH:/opt/tiger/ptx__
