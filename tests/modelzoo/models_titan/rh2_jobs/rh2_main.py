r"""
This is the public entry function of all bale packing rh2 jobs.
"""

import os
import sys
import subprocess
from copy import deepcopy

from .utils.config import cfg, update_cfg
from .rh2_utils import export_titan_cfg

__all__ = ['converter_main']


def converter_main(exec_path, rh2_env, rh2_params_parser, cfg_yaml):
    # Path initialization
    exec_path = os.path.dirname(os.path.realpath(exec_path))
    titan_root_path = os.path.join(exec_path, '../..')
    converting_cfg_path = os.path.join(exec_path, 'converting_config.yaml')

    # Get the original yaml file and load it as adict
    if cfg_yaml != '':
        update_cfg(cfg, cfg_yaml=cfg_yaml, cfg_argv=[], local_rank=0)
    converting_cfg = deepcopy(cfg)

    # Parse rh2 params
    converting_cfg, argv_param = rh2_params_parser(
        converting_cfg=converting_cfg,
        rh2_params=rh2_env.params,
        rh2_inputs=rh2_env.inputs,
        rh2_outputs=rh2_env.outputs)()

    # Export the updated packing config
    export_titan_cfg(converting_cfg, output_path=converting_cfg_path)

    # Run titan model conversion
    for convert_type in cfg.convert_types:
        if convert_type in ['TRT', 'ONNX', 'TorchScript']:
            if convert_type == 'TRT':
                conversion_script = 'convert_TRT.py'
            elif 'bert' not in cfg.model_name:  # bert to onnx not supported
                conversion_script = 'convert_onnx_torchscript.py'
            else:
                conversion_script = 'convert_bert_torchscript.py'

            if conversion_script in ['convert_TRT.py', 'convert_onnx_torchscript.py']:
                cmd = f'cd {titan_root_path} && python3 tools/{conversion_script} ' \
                    f'--model_name {converting_cfg.model_name} ' \
                    f'--model_output_dir {converting_cfg.output_path} '
                if converting_cfg.input_shape:
                    cmd += f'--input_shape {converting_cfg.input_shape} '
                if converting_cfg.pretrained:
                    cmd += '--pretrained '
                    if converting_cfg.tos_bucket:
                        cmd += f'--tos_bucket {converting_cfg.tos_bucket} '
                    if converting_cfg.tos_access_key:
                        cmd += f'--tos_access_key {converting_cfg.tos_access_key} '
                    if converting_cfg.pretrained_version:
                        cmd += f'--model_version {converting_cfg.pretrained_version} '
                    if converting_cfg.pretrained_uri:
                        cmd += f'--model_path {converting_cfg.pretrained_uri} '
                if converting_cfg.verify:
                    cmd += '--verify '
            else:   # convert_bert_torchscript.py
                cmd = f'cd {titan_root_path} && python3 tools/{conversion_script} ' \
                    f'--model_name {converting_cfg.model_name} ' \
                    f'--model_output_dir {converting_cfg.output_path} '
            cmd += f'{argv_param} '
        else:  # Lego conversion
            if 'bert' in cfg.model_name:
                conversion_script = 'convert_bert_lego.py'
            else:
                conversion_script = 'convert_torch_lego.py'
    
            cmd = f'cd {titan_root_path} && python3 tools/{conversion_script} ' \
                  f'-c {converting_cfg.input_path} -w ./tmp ' \
                  f'-o {converting_cfg.output_path} ' \
                  f'--input-shapes [1,{converting_cfg.input_shape}] '
            cmd += f'{argv_param} '

        print(cmd)
        exit_code = subprocess.call(cmd, shell=True, executable='/bin/bash')

        # Exit code return
        sys.exit(exit_code)
