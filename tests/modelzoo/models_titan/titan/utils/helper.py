from .registry import is_model, model_entrypoint, list_all_models
from .hdfs import (
    download_from_hdfs,
    has_hdfs_path_prefix
)
from .misc import (
    download_http_to_local_file,
    has_http_path_prefix
)
from .logger import logger
import timm
import os
import tempfile
import errno
import torch
import torch.distributed as dist
import requests


def is_writable(path):
    try:
        testfile = tempfile.TemporaryFile(dir = path)
        testfile.close()
    except OSError as e:
        if e.errno == errno.EACCES:  # 13
            return False
        e.filename = path
        raise
    return True


def _determine_cache_dir():
    """Use user folder cache by default, if not writable, fallback to ./tmp"""
    user_cache = os.path.expanduser('~/.cache/titan')
    os.makedirs(user_cache, exist_ok=True)
    if is_writable(user_cache):
        return user_cache
    return './tmp'


TMP_DIR = _determine_cache_dir()
SUPPORTED_MODEL_EXTENTIONS = ['.pth', '.npz']
RH2_URL_TEMPLATE = '{}/api/v1/titan_model/tagging/create'


def assure_local_rank(local_rank=None):
    """Try best to get correct local rank"""
    # 1. trust user input
    if local_rank is not None:
        assert isinstance(local_rank, int), f"local_rank provided must be int, given {type(local_rank)}"
        return local_rank
    # 2. trust env
    env_local_rank = os.environ.get('LOCAL_RANK', None)
    if env_local_rank is not None:
        return int(env_local_rank)

    return 0

def assure_rank(rank=None):
    """Try best to get correct global rank"""
    # 1. trust user input
    if rank is not None:
        assert isinstance(rank, int), f"rank provided must be int, given {type(rank)}"
        return rank

    # 2. trust env
    env_rank = os.environ.get('RANK', None)
    if env_rank is not None:
        return int(env_rank)

    # 3. ddp
    if dist.is_initialized():
        return dist.get_rank()

    return 0

def _should_download_to_local(local_rank, rank, rank_zero_only):
    """Util function to determine if current rank should download files"""
    if rank_zero_only:
        if rank > 0:
            return False
        else:
            return True
    else:
        if rank == 0 or local_rank == 0:
            return True
        else:
            return False


def list_models(backend='titan', **kwargs):
    if backend in ['haggs', 'titan']:
        return list_all_models(**kwargs)
    elif backend == 'timm':
        return timm.list_models(**kwargs)
    else:
        raise ValueError(f'Backend:{backend} is not supported.')


def list_pretrained_model_versions(tos_helper, model_name, backend='titan'):
    if backend not in ['haggs', 'titan']:
        logger.error(
            f'Backend:{backend} does not support listing pretrained '
            f'model versions.')
        raise ValueError(
            f'Backend:{backend} does not support listing pretrained '
            f'model versions.')
    tos_prefix = os.path.join(backend, model_name)
    subfolders = list(tos_helper.list_subfolders(tos_prefix))
    return subfolders


def download_model_weights_from_tos(
        tos_helper,
        tos_model_dir,
        dst_dir=TMP_DIR,
        delimiter='/',
        local_rank=None,
        rank=None,
        rank_zero_only=True):
    # download tos files to local
    local_rank = assure_local_rank(local_rank)
    rank = assure_rank(rank)
    should_download = _should_download_to_local(local_rank, rank, rank_zero_only)
    tos_files = list(tos_helper.list_dir(tos_model_dir))
    if len(tos_files) == 0:
        logger.error(
            f'Model path {tos_model_dir} is not found on titan model zoo.')
        raise ValueError(
            f'Model path {tos_model_dir} is not found on titan model zoo.')
    model_path = None
    if tos_model_dir.endswith(delimiter):
        tos_model_dir = tos_model_dir[:-1]
    model_prefix = tos_model_dir.replace(delimiter, '__')
    model_type = tos_model_dir.split(delimiter)[1]

    # TODO(Zhi): Too hacky
    if 'bert' in model_type and 'deberta' not in model_type:
        # for language model, download all files inside the prefix folder
        model_version = tos_model_dir.split(delimiter)[2]
        model_path = os.path.join(dst_dir, model_version)

        for tos_file in tos_files:
            basename = os.path.basename(tos_file)
            if os.path.exists(os.path.join(model_path, basename)):
                continue
            if should_download:  # duplicate but safer
                tos_helper.download_model_from_tos(
                    tos_file, os.path.join(model_path, basename))
    else:
        # for vision model, just download the .pth or .npz weight file
        for tos_file in tos_files:
            # only one model weights file in one folder
            splits = os.path.splitext(tos_file)
            if len(splits) == 2 and splits[1] in SUPPORTED_MODEL_EXTENTIONS:
                basename = os.path.basename(tos_file)
                model_path = os.path.join(
                    dst_dir, model_prefix + '__' + basename)

                if not os.path.exists(model_path):
                    if should_download:  # duplicate but safer
                        tos_helper.download_model_from_tos(
                            tos_file, model_path)
                else:
                    logger.info(
                        f'Model path {model_path} already exists, '
                        f'skip downloading.')
    # if multi-core or distributed env, sync all downloading process
    if dist.is_initialized() and not rank_zero_only:
        dist.barrier()
    if rank_zero_only and rank > 0:
        model_path = None
    return model_path


def download_weights(
        model_name,
        tos_helper=None,
        pretrained_version=None,
        pretrained_uri=None,
        backend='titan',
        local_rank=None,
        rank=None,
        rank_zero_only=True):
    local_rank = assure_local_rank(local_rank)
    rank = assure_rank(rank)
    if rank_zero_only and rank > 0:
        logger.debug(f"[Rank {rank}] Skipping `download_weights` given rank_zero_only=True")
        return None
    should_download = _should_download_to_local(local_rank, rank, rank_zero_only)
    if tos_helper is not None:
        if pretrained_version is None:
            # if pretrained_version, will choose the first model version in
            # alphabetical order from model zoo
            model_versions = list_pretrained_model_versions(
                tos_helper, model_name)
            if model_versions is None or len(model_versions) == 0:
                logger.warning(
                    'Pretrained model not found in model zoo. '
                    'Will skip pretrained model loading.')
                return None
            else:
                pretrained_version = model_versions[0]
        # TODO: transfer timm pretrained models to model zoo
        tos_dir = os.path.join(backend, model_name, pretrained_version)
        model_path = download_model_weights_from_tos(
            tos_helper, tos_dir, local_rank=local_rank,
            rank=rank, rank_zero_only=rank_zero_only)
    elif pretrained_uri:
        # load model from given path, hdfs/http/local path
        if has_hdfs_path_prefix(pretrained_uri):
            # download from hdfs
            basename = os.path.basename(pretrained_uri)
            model_path = os.path.join(TMP_DIR, basename)

            if not os.path.exists(model_path) and should_download:
                download_from_hdfs(pretrained_uri, model_path)
        elif has_http_path_prefix(pretrained_uri):
            # download from http url
            basename = os.path.basename(pretrained_uri)
            model_path = os.path.join(TMP_DIR, basename)

            if not os.path.exists(model_path) and should_download:
                download_http_to_local_file(pretrained_uri, model_path)
        else:
            model_path = pretrained_uri
    else:
        logger.warning(
            'Pretrained model not found in model zoo. '
            'Will skip pretrained model loading.')
        model_path = None
    if rank_zero_only and rank > 0:
        model_path = None
    return model_path


def load_pretrained_model_weights(model, model_path, strict=False, rm_deberta_prefix=False, **kwargs):
    if model_path is None:
        # rank zero only mode, other rank skip loading
        return
    map_location = kwargs.pop('map_location', 'cpu')

    # load weights
    ckpt = torch.load(model_path, map_location=map_location)
    if ckpt.get('state_dict', None):
        state_dict = ckpt['state_dict']
    elif ckpt.get('state_dict_ema', None):
        state_dict = ckpt['state_dict_ema']
    elif ckpt.get('model', None):
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    if rm_deberta_prefix:
        new_state_dict = {}
        for k,v in state_dict.items():
            if k.startswith("deberta.deberta."):
                k = k.replace("deberta.deberta.", "deberta.")
            elif k.startswith("deberta."):
                k = k.replace("deberta.", "")
            new_state_dict[k] = v
        keys = model.load_state_dict(new_state_dict, strict=strict)
    else:
        keys = model.load_state_dict(state_dict, strict=strict)

    if len(keys.missing_keys) > 0:
        logger.warning(
            f"=> Pretrained: missing_keys [{', '.join(keys.missing_keys)}]")
    if len(keys.unexpected_keys) > 0:
        logger.warning(
            f"=> Pretrained: unexpected_keys ["
            f"{', '.join(keys.unexpected_keys)}]")
    logger.info(
        f'=> Pretrained: load pretrained model with strict={strict}')


def create_model(
        model_name,
        pretrained=False,
        pretrained_version=None,
        pretrained_uri=None,
        backend='titan',
        features_only=False,
        local_rank=None,
        rank=None,
        **kwargs):
    if backend in ['haggs', 'titan']:
        if is_model(model_name):
            create_fn = model_entrypoint(model_name)
        else:
            raise RuntimeError('Unknown model (%s)' % model_name)
        model = create_fn(
            pretrained=pretrained,
            features_only=features_only,
            pretrained_version=pretrained_version,
            pretrained_uri=pretrained_uri,
            local_rank=local_rank,
            rank=rank,
            **kwargs)
        weight_name = get_pretrained_weight_name(
            model_name=model_name,
            pretrained=pretrained,
            pretrained_version=pretrained_version,
            pretrained_uri=pretrained_uri,
            **kwargs
        )
        # report model info to rh2
        report_model_info(model_name, weight_name, backend)
    elif backend == 'timm':
        try:
            model = timm.create_model(
                model_name, pretrained, features_only=features_only, **kwargs)
        except Exception as e:
            logger.error(f'Error message: {e}')
            return None
        # report model info to rh2
        report_model_info(model_name, 'unknown', backend)
    else:
        raise ValueError(f'Backend type: {backend} is not supported.')
    return model


def delete_model(tos_helper, model_path):
    logger.warning(f'Delete file {model_path} in TOS.')
    tos_helper.delete(model_path)


def replace_module_weights(
        module,
        tos_helper=None,
        pretrained_version=None,
        pretrained_uri=None,
        backend='titan'):
    weight_path = download_weights(
        module.name,
        tos_helper=tos_helper,
        pretrained_version=pretrained_version,
        pretrained_uri=pretrained_uri,
        backend=backend
    )
    load_pretrained_model_weights(module, weight_path)
    logger.info(f'Module:{module.name} pre-trained loading is done.')


def get_configs(**kwargs):
    pretrained_config = {}
    pretrained_config['tos_helper'] = kwargs.pop('tos_helper', None)
    pretrained_config['pretrained_version'] = kwargs.pop('pretrained_version', None)
    pretrained_config['pretrained_uri'] = kwargs.pop('pretrained_uri', None)
    pretrained_config['local_rank'] = kwargs.pop('local_rank', None)
    pretrained_config['rank'] = kwargs.pop('rank', None)
    pretrained_config['rank_zero_only'] = kwargs.pop('rank_zero_only', True)
    pretrained_config['backend'] = kwargs.pop('backend', 'titan')

    model_config = kwargs.copy()
    if kwargs.get('features_only', None) is None:
        model_config['features_only'] = False

    return pretrained_config, model_config


def get_pretrained_weight_name(
        model_name,
        pretrained=False,
        pretrained_version=None,
        pretrained_uri=None,
        **kwargs):
    weight_name = 'none'
    if pretrained:
        tos_helper = kwargs.get('tos_helper', None)
        if tos_helper is not None:
            if pretrained_version is not None:
                # if pretrained_version, will choose the first model version in
                # alphabetical order from model zoo
                model_versions = list_pretrained_model_versions(
                    tos_helper, model_name)
                if model_versions is None and len(model_versions) > 0:
                    weight_name = model_versions[0]
        elif pretrained_uri and (has_hdfs_path_prefix(pretrained_uri) or
                has_http_path_prefix(pretrained_uri)):
            weight_name = pretrained_uri
    return weight_name


def report_model_info(model_name, weight_name, backend='titan', rank=None):
    rank = assure_rank(rank=rank)
    if rank > 0:
        logger.debug(f"[Rank {rank}] Skipping `report_model_info`")
        return
    content = {}
    content['model_name'] = model_name
    content['weight_name'] = weight_name
    content['backend'] = backend
    content['job_run_id'] = os.getenv('RH2_JOB_RUN_ID', 'unknown')
    extra = []
    extra.append({'key': 'arnold_monitor_task_owner', 'value': os.getenv('ARNOLD_MONITOR_TASK_OWNER', 'unknown')})
    extra.append({'key': 'arnold_monitor_trial_owner', 'value': os.getenv('ARNOLD_MONITOR_TRIAL_OWNER', 'unknown')})
    extra.append({'key': 'arnold_job_id', 'value': os.getenv('ARNOLD_JOB_ID', 'unknown')})
    extra.append({'key': 'arnold_task_id', 'value': os.getenv('ARNOLD_TASK_ID', 'unknown')})
    extra.append({'key': 'arnold_trial_id', 'value': os.getenv('ARNOLD_TRIAL_ID', 'unknown')})
    extra.append({'key': 'arnold_monitor_rh2_job_def_name', 'value': os.getenv('ARNOLD_MONITOR_RH2_JOB_DEF_NAME', 'unknown')})
    content['extra'] = extra

    region = os.getenv('ARNOLD_REGION')
    rh2_host = ''
    if region == 'CN':
        rh2_host = 'https://rh2.bytedance.net'
    elif region == 'US':
        rh2_host = 'https://rh2-va.bytedance.net'
    else:
        logger.warning(f'unknown region:{region}')
        return
    url = RH2_URL_TEMPLATE.format(rh2_host)

    try:
        resp = requests.post(url=url, json=content)
        if resp.ok:
            logger.info('Successfully send post request to rh2 to report model info.')
        else:
            logger.error(f'Failed to send post request to rh2 with status_code:{resp.status_code}')
    except Exception as e:
        logger.error(f'Failed to report model info to rh2, error: {e}.')
