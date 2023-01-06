import hashlib
import os
import torch

from typing import Optional, Union

EASYGUARD_CACHE = os.path.join(f"{os.environ['HOME']}/.cache", "easyguard")
EASYGUARD_MODEL_CACHE = os.path.join(EASYGUARD_CACHE, "models")
REMOTE_PATH_SEP = "/"

from . import hmget, hexists, file_read
from .logging import get_logger
from ..modelzoo.config import MODEL_ARCHIVE_PATH

logger = get_logger(__name__)


def file_exist(prefix: str, choices: Union[str, set]) -> Optional[str]:
    """check

    Parameters
    ----------
    prefix : str
        _description_
    choices : Union[str, set]
        _description_

    Returns
    -------
    Optional[str]
        _description_
    """
    if isinstance(choices, str):
        path_ = os.path.join(prefix, choices)
        if os.path.exists(path_):
            return path_
    else:
        for choice in choices:
            path_ = os.path.join(prefix, choice)
            if os.path.exists(path_):
                return path_
    return None


def sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def cache_file(
    model_name_path: str, file_name: Optional[Union[str, set]] = None, *args, **kwargs
) -> str:
    # TODO (junwei.Dong): 未来支持更多方式的读取
    """支持三种方式读取:
        1.本地指定文件读取: model_name_path直接指定到具体的文件
        2.本地指定模型名字读取: model_name_path + file_name, 举例: deberta_base_6l + config.yaml
        3.服务器端读取: model_name_path + file_name + remote_url, 举例: deberta_base_6l + config.yaml + hdfs://xxx/config.yaml

    Parameters
    ----------
    model_name_path : str
        模型名字或者指定路径
    file_name : Optional[Union[str, set]], optional
        想要获取的文件,按需指定, by default None

    Returns
    -------
    str
        本地文件路径或者远程拉取下来的文件的本地路径

    Raises
    ------
    FileExistsError
        _description_
    NotImplementedError
        _description_
    """
    model_type = kwargs.pop("model_type", None)
    if os.path.exists(model_name_path):
        return model_name_path
    elif model_type is not None:
        _hash = sha256(model_name_path)
        model_path_local = os.path.join(EASYGUARD_MODEL_CACHE, model_type, _hash)
        model_file_local = file_exist(model_path_local, file_name)
        if model_file_local:
            logger.warning(f"obtain the local file `{model_file_local}`")
            return model_file_local
        else:
            # TODO (junwei.Dong): 如果本地不存在那么需要根据url去远程获取，然后放置在特定的缓存目录下, 现目前只支持hdfs
            model_path_remote = kwargs.pop("remote_url", None)
            assert (
                model_path_remote is not None
            ), f"the argument `remote_url` doesn't exist"
            os.makedirs(model_path_local, exist_ok=True)
            if isinstance(file_name, str):
                model_file_remote_list = [
                    REMOTE_PATH_SEP.join([model_path_remote, file_name])
                ]
            else:
                model_file_remote_list = list(
                    map(
                        lambda x: REMOTE_PATH_SEP.join([model_path_remote, x]),
                        file_name,
                    )
                )
            # diffent servers, different processes
            if model_path_remote.startswith("hdfs://"):
                model_file_path_remote = None
                for remote_path_ in model_file_remote_list:
                    if hexists(remote_path_):
                        model_file_path_remote = remote_path_
                        break
                if model_file_path_remote:
                    logger.warning(f"start to download `{model_file_path_remote}`")
                    hmget([model_file_path_remote], model_path_local)
                    # whether the request is successful or not, the `model_file_local` will be created, so we need to check the target file
                    model_file_path_local = os.path.join(
                        model_path_local,
                        model_file_path_remote.split(REMOTE_PATH_SEP)[-1],
                    )
                    if os.path.getsize(model_file_path_local) == 0:
                        os.remove(model_file_path_local)
                        logger.warning(
                            f"fail to download `{model_file_path_remote}`, please check the network or the remote file path"
                        )
                else:
                    raise FileNotFoundError(
                        f"`{model_file_remote_list}` can not find in remote server"
                    )

            final_file_path = file_exist(model_path_local, file_name)
            if final_file_path:
                return final_file_path
            raise FileExistsError(
                f"failed to obtain one of the remote files `{model_file_remote_list}`"
            )
            ...

    else:
        # TODO (junwei.Dong): 不是模型的文件该如何从远程获取并存放在缓存目录
        raise NotImplementedError(f"Just support model cache")
    ...


def list_pretrained_models():
    """list all pretrained models from archive.yaml"""
    from prettytable import PrettyTable

    model_archive = file_read(MODEL_ARCHIVE_PATH)
    model_archive_table = PrettyTable()
    filed_names = list()
    for key_, value_ in model_archive.items():
        filed_names += list(value_.keys())
    filed_names = list(set(filed_names))
    model_archive_table.field_names = ["model_name"] + filed_names
    for key_, value_ in model_archive.items():
        temp_ = [key_] + [value_.get(_, None) for _ in filed_names]
        model_archive_table.add_row(temp_)
    logger.info(model_archive_table)


# from titan
def get_configs(**kwargs):
    pretrained_config = {}
    pretrained_config["tos_helper"] = kwargs.pop("tos_helper", None)
    pretrained_config["pretrained_version"] = kwargs.pop("pretrained_version", None)
    pretrained_config["pretrained_uri"] = kwargs.pop("pretrained_uri", None)
    pretrained_config["local_rank"] = kwargs.pop("local_rank", None)
    pretrained_config["rank"] = kwargs.pop("rank", None)
    pretrained_config["rank_zero_only"] = kwargs.pop("rank_zero_only", True)
    pretrained_config["backend"] = kwargs.pop("backend", "titan")

    model_config = kwargs.copy()
    if kwargs.get("features_only", None) is None:
        model_config["features_only"] = False

    return pretrained_config, model_config


# from titan
def load_pretrained_model_weights(
    model, model_path, strict=False, rm_deberta_prefix=False, **kwargs
):
    """load pretrained model weights

    Parameters
    ----------
    model : _type_
        _description_
    model_path : _type_
        _description_
    strict : bool, optional
        _description_, by default False
    rm_deberta_prefix : bool, optional
        _description_, by default False
    """
    if model_path is None:
        # rank zero only mode, other rank skip loading
        return
    map_location = kwargs.pop("map_location", "cpu")

    # load weights
    ckpt = torch.load(model_path, map_location=map_location)
    if ckpt.get("state_dict", None):
        state_dict = ckpt["state_dict"]
    elif ckpt.get("state_dict_ema", None):
        state_dict = ckpt["state_dict_ema"]
    elif ckpt.get("model", None):
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    if rm_deberta_prefix:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("deberta.deberta."):
                k = k.replace("deberta.deberta.", "deberta.")
            elif k.startswith("deberta."):
                k = k.replace("deberta.", "")
            new_state_dict[k] = v
        keys = model.load_state_dict(new_state_dict, strict=strict)
    else:
        keys = model.load_state_dict(state_dict, strict=strict)

    if len(keys.missing_keys) > 0:
        logger.warning(f"=> Pretrained: missing_keys [{', '.join(keys.missing_keys)}]")
    if len(keys.unexpected_keys) > 0:
        logger.warning(
            f"=> Pretrained: unexpected_keys [" f"{', '.join(keys.unexpected_keys)}]"
        )
    logger.info(f"=> Pretrained: load pretrained model with strict={strict}")
