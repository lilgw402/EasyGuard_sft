# export http_proxy=10.20.47.147:3128
# export https_proxy=10.20.47.147:3128
# export no_proxy=code.byted.org

export http_proxy=http://10.20.47.147:3128 https_proxy=https://10.20.47.147:3128 export no_proxy=code.byted.org

python3 run_model.py --config default_config.yaml

unset http_proxy
unset https_proxy
