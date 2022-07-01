import warnings
import os
from fex.config import CfgNode


def init_tracking(project: str, cfg: CfgNode):
    try:
        import tracking as tk
    except ImportError:
        warnings.warn('Tracking not found. Please install by: pip install byted-tracking')

    if int(os.getenv('RANK', 0)) == 0:
        config_dict = {}
        for key in cfg.flat_keys():
            val = cfg.get(key)
            # only track config of basic type
            if isinstance(val, (int, float, bool, str)):
                config_dict[key] = val
        tk.init(project=project, config=config_dict)
