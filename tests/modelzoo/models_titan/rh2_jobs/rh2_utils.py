"""
The utils to connect Rh2 params and cfg conveniently.
"""

import re
import yaml
import pprint
from addict import Dict as adict

from titan.utils import logger
from .utils.config import add_quotation_to_string

__all__ = [
    'export_titan_cfg',
    'Rh2ConvertingParamsParser',
    'str_to_list',
    'argv_to_adict'
]

RH2_DICT_ROOT = '/opt/tiger/rh2_params'


def export_titan_cfg(bale_cfg, output_path, verbose=True):
    r""" Export the bale cfg updated by rh2 params as a yaml file"""
    with open(output_path, 'w') as f:
        yaml.dump(bale_cfg.to_dict(), f, default_flow_style=False)
    if verbose:
        logger.info(f'\n=============== config json for Rh2 job trial ===============')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(bale_cfg)


def str_to_list(x, force_list=True, default=None):
    r""" Generate bale params from string with format like Union[int, List[int]]

    Args:
        x: string extracted from the rh2 params
        force_list: return a list anyway even if there is only on element in the list
        default: the legal value
    """

    assert isinstance(x, str), f'{x} must be a string, find {type(x)} instead'
    if x == '':
        return default
    x_list = eval(f'[{add_quotation_to_string(x)}]')

    if force_list is False and len(x_list) == 1:
        return x_list[0]
    else:
        return x_list


def argv_to_adict(x, prefix):
    r""" Process the argv format params and return an adict """
    assert isinstance(x, str), f'{x}, {type(x)}'
    x_dict = adict()
    for element in re.split(r'\s', x):
        if element == '' or not element.startswith(prefix):
            continue
        assert len(element.split('=')) == 2, f'Illegal argv format {element}'
        key, value = element.split('=')
        key = key[len(prefix) + 1:]
        try:
            exec(f'x_dict.{key}={value}')
        except NameError:
            exec(f'x_dict.{key}="{value}"')
    return x_dict


class Rh2ConvertingParamsParser(object):
    def __init__(self, converting_cfg, rh2_params, rh2_inputs, rh2_outputs):
        self.cfg = converting_cfg
        self.rh2 = rh2_params
        self.rh2_inputs = rh2_inputs
        self.rh2_outputs = rh2_outputs
        self.argv = ''

    def common_params_update(self):
        self.cfg.model_name = self.rh2.model_name
        self.cfg.pretrained = self.rh2.pretrained
        if self.cfg.pretrained:
            self.cfg.backend = self.rh2.backend
            self.cfg.tos_bucket = self.rh2.tos_bucket
            self.cfg.tos_access_key = self.rh2.tos_access_key
            if self.rh2.pretrained_version != '':
                self.cfg.pretrained_version = self.rh2.pretrained_version
            if self.rh2.pretrained_uri != '':
                self.cfg.pretrained_uri = self.rh2.pretrained_uri
        self.cfg.input_shape = self.rh2.input_shape
        self.cfg.convert_types = self.rh2.convert_types
        self.cfg.input_path = self.rh2.input_path
        self.cfg.output_path = self.rh2.output_path
        self.cfg.verify = self.rh2.verify

    def task_params_update(self):
        r""" The task related params update with highest priority """
        pass

    def __call__(self):
        # update the common settings
        self.common_params_update()

        # update the task unique settings
        self.task_params_update()

        return self.cfg, self.argv
