import os
from typing import Optional

from ..utils import (
    EASYGUARD_CACHE,
    EASYGUARD_MODEL_CACHE,
    REMOTE_PATH_SEP,
    sha256,
)


class BaseAutoHubClass:
    ...

    @classmethod
    def from_name(cls, server_name):
        ...


# TODO (junwei.Dong): provide hdfs download capability for each general module, such as tokenizebase, modelbase, configurationbase and etc.
class HdfsHub:
    def __init__(
        self, model_name: str, model_dir_remote: str, model_type: str
    ) -> None:
        ...
        self._model_dir_remote = model_dir_remote
        self._model_dir_local = os.path.join(
            EASYGUARD_MODEL_CACHE, model_type, sha256(model_name)
        )

    def get_file(self, file_name: str, force: Optional[bool] = False):
        file_path_local = os.path.join(self._model_dir_local, file_name)
        file_path_remote = os.path.join(self._model_dir_remote, file_name)
        if self.download_file(file_path_remote, self._model_dir_local, force):
            return file_path_local
        raise FileExistsError(
            f"the {file_name} does not exist locally and remotely, please check~"
        )

    @classmethod
    def download_file(
        cls, hdfs_url: str, target_dir: str, force: Optional[bool] = False
    ) -> bool:
        ...
