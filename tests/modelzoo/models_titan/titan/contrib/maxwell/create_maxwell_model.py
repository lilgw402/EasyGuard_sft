
import sys
from pathlib import Path
from titan.utils.misc import download_and_extract_scm_permission
from torch.distributed import is_initialized, barrier

def create_maxwell_model(conf_pth, local_rank=0):
    exist_maxwell = any(path.endswith('maxwell') for path in sys.path)
    maxwell_dest = '/opt/tiger/maxwell'
    if not exist_maxwell:
        if not Path(maxwell_dest).exists() and local_rank == 0:
            # Download SCM
            download_and_extract_scm_permission(maxwell_dest, 'data.content_security.maxwell', '1.0.0.1038')
        if is_initialized():
            barrier()
    
    sys.path.insert(0, maxwell_dest)
    from models import build_model
    from addict import Dict as ADict
    from utils.util import load_conf
    from utils.config import config

    conf = ADict(load_conf(conf_pth))
    config.update(conf)
    model_instance_conf = ADict(config.model_instance)
    model_type = model_instance_conf.pop("type")
    if config.QAT.qat_flag == True:
        from feather.quant.model_surgeon import create_model_surgeon
        with create_model_surgeon(backend='trt'):
            model = build_model(model_type, model_instance_conf)
    else:
        model = build_model(model_type, model_instance_conf)

    return model
