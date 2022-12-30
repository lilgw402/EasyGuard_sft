import os
import yaml
import tempfile
from typing import List, Tuple
from addict import Dict as adict

from titan.utils import logger
from titan.utils.misc import (
    download_http_to_local_file,
    is_number_or_bool_or_none,
    force_create_empty_folder
)
from titan.utils.hdfs import (
    is_hdfs_dir,
    hdfs_mkdir,
    is_hdfs_file,
    has_hdfs_path_prefix,
    download_from_hdfs
)

SNAPSHOT_CFG_PATH = './snapshot.yaml'
GLOBAL_CFG_PATH = './global.yaml'


cfg = adict()

# Common cfg
cfg.model_name = ''
cfg.pretrained = False
cfg.backend = 'titan'
cfg.tos_bucket = None
cfg.tos_access_key = None
cfg.pretrained_version = None
cfg.pretrained_uri = None
cfg.input_shape = None
cfg.convert_types = ['TRT']
cfg.input_path = None
cfg.output_path = None
cfg.verify = False


def add_quotation_to_string(s: str,
                            split_chars: List[str] = None) -> str:
    r""" For eval() to work properly, all string must be added quatation.
         Example: '[[train,3],[val,1]' -> '[["train",3],["val",1]'

    Args:
        s: the original value string
        split_chars: the chars that mark the split of the string

    Returns:
        the quoted value string
    """
    if split_chars is None:
        split_chars = ['[', ']', '{', '}', ',', ' ']
        if '{' in s and '}' in s:
            split_chars.append(':')
    s_mark, marker = s, chr(1)
    for split_char in split_chars:
        s_mark = s_mark.replace(split_char, marker)
    s_quoted = ''
    for value in s_mark.split(marker):
        if len(value) == 0:
            continue
        st = s.find(value)
        if is_number_or_bool_or_none(value):
            s_quoted += s[:st] + value
        elif value.startswith("'") and value.endswith("'") or value.startswith('"') and value.endswith('"'):
            s_quoted += s[:st] + value
        else:
            s_quoted += s[:st] + '"' + value + '"'
        s = s[st + len(value):]
    return s_quoted + s


# Update cfg from agrv for override
def update_cfg_from_argv(cfg: adict,
                         cfg_argv: List[str],
                         delimiter: str = '=') -> None:
    r""" Update global cfg with list from argparser

    Args:
        cfg: the cfg to be updated by the argv
        cfg_argv: the new config list, like ['epoch=10', 'save.last=False']
        dilimeter: the dilimeter between key and value of the given config
    """

    def resolve_cfg_with_legality_check(keys: List[str]) -> Tuple[adict, str]:
        r""" Resolve the parent and leaf from given keys and check their legality.

        Args:
            keys: The hierarchical keys of global cfg

        Returns:
            the resolved parent adict obj and its legal key to be upated.
        """

        obj, obj_repr = cfg, 'cfg'
        for idx, sub_key in enumerate(keys):
            if not isinstance(obj, adict) or sub_key not in obj:
                raise ValueError(f'Undefined attribute "{sub_key}" detected for "{obj_repr}"')
            if idx < len(keys) - 1:
                obj = obj.get(sub_key)
                obj_repr += f'.{sub_key}'
        return obj, sub_key

    for str_argv in cfg_argv:
        item = str_argv.split(delimiter, 1)
        assert len(item) == 2, "Error argv (must be key=value): " + str_argv
        key, value = item
        obj, leaf = resolve_cfg_with_legality_check(key.split('.'))
        obj[leaf] = eval(add_quotation_to_string(value))


def _recursive_update(_cfg, _user_cfg):
    r""" Recursively merge user cfg into original cfg

    :param _cfg: The original cfg to be updated.
    :param _user_cfg: The user config given from yaml file.
    """

    # Basic legality check: the original config
    assert isinstance(_cfg, dict), \
        f'Merge user cfg into original cfg fails since cfg {_cfg} is not a dict'
    assert isinstance(_user_cfg, dict), \
        f'Merge user cfg into original cfg fails since user cfg {_user_cfg} is not a dict'

    # Traverse through the user cfg
    for key, value in _user_cfg.items():
        assert key in _cfg, \
            f'Key is restricted among {list(_cfg.keys())}, got {key} instead.'
        # Recursive expand only when _cfg[key] is adict()
        if isinstance(_cfg[key], adict):
            _recursive_update(_cfg[key], value)
        # Directly update the _cfg[key] otherwise
        else:
            if isinstance(value, str) and value.strip().lower() in ['none', 'null']:
                value = None
            _cfg[key] = value


def update_cfg(cfg: adict,
               cfg_yaml: str,
               cfg_argv: List[str],
               local_rank: int) -> None:
    r""" Update cfg with user yaml file and agrv

    Args:
        cfg: the original global cfg to be updated
        cfg_yaml: the user cfg yaml path, local/online/hdfs paths are supported
        cfg_argv: the supplimentary argv from command line
        local_rank: identifier to avoid download conflict on different locak_rank

    Returns:
        the updated cfg
    """
    # Update hfds root
    if cfg.hdfs.root is None:
        arnold_base_dir = os.environ.get('ARNOLD_BASE_DIR', '')
        if arnold_base_dir.startswith('hdfs://harunava'):
            cfg.hdfs.root = cfg.hdfs.harunava
        elif arnold_base_dir.startswith('hdfs://haruna'):
            cfg.hdfs.root = cfg.hdfs.haruna
        else:
            logger.warning(f'Unrecognized ARNOLD_BASE_DIR: {arnold_base_dir}')

    with tempfile.TemporaryDirectory(prefix='cfg_yaml') as temp_cfg_yaml_dir:
        # Download cfg_yaml file from hdfs/tos if necessary.
        if cfg_yaml.startswith('http'):  # download yaml file from TOS
            download_path = os.path.join(temp_cfg_yaml_dir, 'tos.yaml')
            if os.path.exists(download_path):
                os.remove(download_path)
            download_http_to_local_file(cfg_yaml, download_path)
            cfg_yaml = download_path
        elif not os.path.exists(cfg_yaml):  # download yaml file from hdfs
            # yaml file existance check
            if not cfg_yaml.startswith('hdfs'):
                cfg_yaml_hdfs = os.path.join(cfg.hdfs.root, cfg_yaml)
                assert is_hdfs_file(cfg_yaml_hdfs), \
                    f'{cfg_yaml_hdfs} does not exist on hdfs; Moreover, ' \
                    f'{cfg_yaml} does not exist at local disk as well'
                cfg_yaml = cfg_yaml_hdfs
            else:
                assert is_hdfs_file(cfg_yaml), f'{cfg_yaml} does not exist'
            # download yaml file from hdfs
            download_path = os.path.join(temp_cfg_yaml_dir, os.path.basename(cfg_yaml))
            if os.path.exists(download_path):
                os.remove(download_path)
            download_from_hdfs(src_path=cfg_yaml, dst_path=download_path)
            cfg_yaml = download_path

        # Update default cfg with given yaml file
        with open(cfg_yaml) as f:
            user_cfg = yaml.load(f, Loader=yaml.FullLoader)
            _recursive_update(cfg, user_cfg)

        # Make sure the global config is adict and update cfg from given argv list
        for key, value in cfg.items():
            cfg[key] = adict(tmp=value)['tmp']
        update_cfg_from_argv(cfg, cfg_argv)


def update_converting_cfg(cfg: adict,
                          cfg_yaml: str,
                          cfg_argv: List[str],
                          local_rank: int) -> adict:
    r""" Update cfg with user yaml file and agrv

    Args:
        cfg: the original global packing cfg
        cfg_yaml: the user cfg yaml path, local/online/hdfs paths are supported
        cfg_argv: the supplimentary argv from command line
        local_rank: identifier to avoid download conflict on different locak_rank

    Returns:
        the updated cfg
    """
    update_cfg(cfg, cfg_yaml, cfg_argv, local_rank)

    # Output_path preprocess
    if has_hdfs_path_prefix(cfg.output_path):
        if not is_hdfs_dir(cfg.output_path):
            hdfs_mkdir(cfg.output_path, raise_exception=True)
    else:
        force_create_empty_folder(cfg.output_path)
