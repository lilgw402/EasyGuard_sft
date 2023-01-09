r""" Miscellaneous file to provide some basic tools.
     Some are copied from
     [haggs](https://code.byted.org/lab/haggs/blob/master/haggs/utils/misc.py).
"""
import os
import requests
import json
import subprocess
from pathlib import Path

from .logger import logger

__all__ = [
    'is_number_or_bool_or_none',
    'download',
    'download_http_to_local_file',
    'download_and_extract_scm',
    'download_and_extract_scm_permission',
    'get_file_extension',
    'check_tos_model_path',
    'has_http_path_prefix',
    'get_lang_config_file',
    'get_lang_checkpoint_file',
    'load_json_config',
    'force_create_empty_folder'
]

# supported HDFS FS prefix
_SUPPORTED_HTTP_PATH_PREFIXES = ('http://', 'https://')


def is_number_or_bool_or_none(x):
    r""" Return True if the given str represents a number (int or float) or bool
    """

    try:
        float(x)
        return True
    except ValueError:
        return x in ['True', 'False', 'None']


def download(url, timeout=20, retry=3):
    finish_download = False
    time_to_try = retry
    content = None
    while time_to_try and not finish_download:
        rsp = requests.get(url, timeout=timeout)
        if rsp.status_code == requests.codes.ok:
            finish_download = True
            content = rsp.content
        else:
            time_to_try -= 1
    if finish_download and content is not None:
        return content
    else:
        raise Exception('download {} failed'.format(url))


def download_http_to_local_file(url, local_path, timeout=20, retry=3):
    try:
        fd = os.open(local_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        content = download(url, timeout=timeout, retry=retry)
        os.write(fd, content)
        os.close(fd)
    except:
        logger.info(
            f'Model path {local_path} already exists, '
            f'skip downloading.')


def get_file_extension(filepath):
    """
    Get file extension.
    :param filepath: str, file path.
    :return extension: str.
    """
    return filepath.split('/')[-1].split('.', 1)[-1]


def check_tos_model_path(model_path, delimiter='/'):
    """
    Check if model path is valid.
    Model path conversion: {backend}/{model_name}/{model_version}/{file_name}

    :param model_path: str, model path on tos.
    :return valid: bool.
    """
    subpaths = model_path.split(delimiter)
    if not subpaths[0] in ['haggs', 'timm'] and len(subpaths) != 4:
        return False
    return True


def has_http_path_prefix(filepath):
    """
    Check if filepath is a valid http url.

    :param model_path: str.
    :return valid: bool.
    """
    for prefix in _SUPPORTED_HTTP_PATH_PREFIXES:
        if filepath.startswith(prefix):
            return True
    return False


def get_lang_config_file(base_dir):
    """
    Find config file of language model from the base directory

    :param base_dir: base directory.
    :return config_file: full path of config file. If not found, return None.
    """
    config_file = None
    # default config file name is 'config.json'
    if os.path.exists(os.path.join(base_dir, 'config.json')):
        return os.path.join(base_dir, 'config.json')

    files = os.listdir(base_dir)
    for f in files:
        if get_file_extension(f) == 'json':
            return os.path.join(base_dir, f)
    return config_file


def get_lang_checkpoint_file(base_dir):
    """
    Find checkpoint file of language model from the base directory

    :param base_dir: base directory.
    :return weight_file: full path of ckpt file. If not found, return None.
    """
    files = os.listdir(base_dir)
    for f in files:
        if get_file_extension(f) == 'bin':
            return os.path.join(base_dir, f)
    raise IOError(f'No checkpoint file is found in {base_dir}.')


def load_json_config(json_path):
    """
    Load json config into a dict

    :param json_path: path to input json file.
    :return config_dict: dict of all configs
    """
    if not os.path.exists(json_path):
        raise IOError(f'json_path:{json_path} does not exists.')

    with open(json_path) as f:
        config_dict = json.load(f)
    return config_dict


def download_and_extract_scm(local_path, scm: str, version):
    if '/' in scm:
        scm.replace('/', '.')
    download_url = f'http://d.scm.byted.org/api/v2/download/ceph:{scm}_{version}.tar.gz'
    destination = str(Path(local_path).parent.joinpath(f'{scm}.tar.gz'))
    if os.path.exists(destination):
        return
    download_http_to_local_file(download_url, destination)
    os.mkdir(local_path)
    subprocess.call(['tar', '-zvxf', destination, '-C', local_path], stdout=subprocess.DEVNULL)
    os.remove(destination)


def get_scm_token():
    headers = {"authorization": "Bearer 7cd4a64b5fc07c659a5980a984d6dd64"}
    req = requests.get("http://cloud.bytedance.net/auth/api/v1/jwt", headers=headers)
    jwt_token = req.headers.get("X-Jwt-Token")
    return jwt_token


def download_and_extract_scm_permission(local_path, scm: str, version):
    """
    SCM permission is restricted after 2022.7.15. This is the new API for downloading
    """
    scm_token = get_scm_token()
    scm_host = 'luban-source.byted.org'

    scm_cmd = "wget -O {}_{}.tar.gz http://{}/repository/scm/{}_{}.tar.gz --header='x-jwt-token: {}' --header='x-platform-proxy-user: {}' "
    destination = f"{scm}_{version}.tar.gz"
    if os.path.exists(destination):
        return
    cmd = scm_cmd.format(
            scm, version, scm_host, scm, version, scm_token, os.environ['ARNOLD_TRIAL_OWNER'])
    pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    res = pipe.communicate()
    if pipe.returncode:
        err_msg = res[1]
        print("Get SCM {}:{} fail, err is {}".format(scm, version, err_msg))
        success = False
    else:
        stdout_msg = res[0]
        print("Get SCM {}:{}, stdout is {}".format(scm, version, stdout_msg))
    
    # os.mkdir(local_path)
    os.mkdir(local_path)
    subprocess.call(['tar', '-zvxf', destination, '-C', local_path], stdout=subprocess.DEVNULL)
    os.remove(destination)

def force_create_empty_folder(path, exist_ok=True):
    r"""
    Establish an empty folder, if the path already exist, remove the existing file/folder first.
    """

    assert isinstance(path, str), f'Illegal path {path} with type {type(path)}'
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=exist_ok)
