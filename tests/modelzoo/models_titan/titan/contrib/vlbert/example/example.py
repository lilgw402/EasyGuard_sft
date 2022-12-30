# flake8: noqa
# pylint: disable=missing-function-docstring, line-too-long, unused-argument

import sys
from pathlib import Path
import argparse
from titan.utils.misc import download_and_extract_scm_permission


exist_vlbert = any('vlbert' in path for path in sys.path)
if not exist_vlbert:
    # requires vlbert: https://code.byted.org/shicheng.cecil/vlbert
    vlbert_dest = '/opt/tiger/vlbert'
    if not Path(vlbert_dest).exists():
        # Download SCM
        download_and_extract_scm_permission(vlbert_dest, 'lab.moderation.vlbert', '1.0.0.183')
    sys.path.insert(0, vlbert_dest)

sys.path.append('/opt/tiger/vlbert/')
from haggs.utils.config import cfg, update_cfg
from titan.contrib.vlbert.create_vlbert_model import create_vlbert_model
from haggs.utils import build_from_cfg, ENV


# parse vlbert config file 
def parse_args(cfg_pth):
    parser = argparse.ArgumentParser(description='Haggs')
    # General
    parser.add_argument('--cfg', type=str, default=cfg_pth,
                        help='experiment configure file name')
    # See https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py
    parser.add_argument("--local_rank", type=int, default=0)

    args, cfg_overrided = parser.parse_known_args()
    # Update config from yaml and argv for override
    update_cfg(args.cfg, cfg_overrided, args.local_rank)

    return args



# create model
conf_pth='/opt/tiger/titan/titan/contrib/vlbert/example/example.yaml'
args = parse_args(cfg_pth=conf_pth)
ENV.local_rank = 0
ENV.cfg = cfg

model=create_vlbert_model(cfg)
print(model)

