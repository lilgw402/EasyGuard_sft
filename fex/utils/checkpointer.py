# -*- coding: utf-8 -*-
'''
Created on Nov-13-20 16:50
checkpointer.py
@author: liuzhen.nlp
Description:
'''

from typing import Union, Dict, List, Tuple, Any, Callable
import logging
import os
import re
import time

import torch

from fex.utils.hdfs_io import hexists, hmkdir, hcopy, hlist_files, hopen, hrm
from fex.utils.torch_io import save as hdfs_torch_save
logger = logging.getLogger(__name__)


class Checkpointer:
    """
    这个类主要是将training checkpointer和state存储到hdfs上.
    """

    def __init__(self,
                 serialization_dir: str = ".output",
                 keep_serialized_model_every_num_seconds: int = None,
                 num_serialized_models_to_keep: int = 4) -> None:
        self._serialization_dir = serialization_dir
        self._keep_serialized_model_every_num_seconds = keep_serialized_model_every_num_seconds
        self._num_serialized_models_to_keep = num_serialized_models_to_keep  # 这个参数暂时没用
        if not hexists(self._serialization_dir):  # TODO: 加一个锁
            hmkdir(self._serialization_dir)

        self._last_permanent_saved_checkpoint_time = time.time()
        self._serialized_paths: List[Tuple[str, str]] = []

    def save_checkpoint(self,
                        epoch: Union[int, str],
                        model_state: Dict[str, Any],
                        training_states: Dict[str, Any],
                        is_best_so_far: bool = False) -> None:
        """
        保存 checkpoint到本地local和remote hdfs中：
        args:
            epoch: 当前训练的epoch数
            model_state: 当前训练model的参数
            training_states: 当前训练的参数
            is_best_so_far: 当前是否save的checkpoint是否为最优
        """
        if self._serialization_dir is not None:
            model_path = os.path.join(
                self._serialization_dir, "model_state_epoch_{}.th".format(epoch))
            training_path = os.path.join(self._serialization_dir,
                                         "middle_training_state_{}".format(epoch))
            logger.info("save ckpt to %s" % model_path)
            hdfs_torch_save(model_state, model_path)
            hdfs_torch_save(training_states, training_path)
            # should write success flag in case checkpoint is incomplete
            flag_path = os.path.join(self._serialization_dir, "success_{}".format(epoch))
            with hopen(flag_path, "w"):
                pass

            if is_best_so_far:
                logger.info("Best validation performance so far. "
                            "Copying weights to '%s/best.th'.", self._serialization_dir)
                hcopy(model_path, os.path.join(
                    self._serialization_dir, "best.th"))

            if self._num_serialized_models_to_keep and self._num_serialized_models_to_keep >= 0:
                self._serialized_paths.append((model_path, training_path))
                if len(self._serialized_paths) > self._num_serialized_models_to_keep:
                    paths_to_remove = self._serialized_paths.pop(0)
                    # Check to see if we should keep this checkpoint, if it has been longer
                    # then self._keep_serialized_model_every_num_seconds since the last
                    # kept checkpoint.
                    remove_path = True
                    if self._keep_serialized_model_every_num_seconds is not None:
                        save_time = paths_to_remove[0]
                        time_since_checkpoint_kept = save_time - \
                            self._last_permanent_saved_checkpoint_time
                        if time_since_checkpoint_kept > self._keep_serialized_model_every_num_seconds:
                            # We want to keep this checkpoint.
                            remove_path = False
                            self._last_permanent_saved_checkpoint_time = save_time

                    if remove_path:
                        hrm(paths_to_remove[1])  # 1 for 只删除 training state，模型参数保留

    @staticmethod
    def find_latest_checkpoint_from_dir(ckpt_dir):
        """
        Return the location of the latest model and training state files.
        If there isn't a valid checkpoint then return None.
        """
        if ckpt_dir is None or not hexists(ckpt_dir):
            return None

        checkpoint_flags = [
            x for x in hlist_files([ckpt_dir]) if "success_" in x]
        if not checkpoint_flags:
            return None
        # Get the last checkpoint file.  Epochs are specified as either an
        # int (for end of epoch files) or with epoch and timestamp for
        # within epoch checkpoints, e.g. 5.2018-02-02-15-33-42
        found_epochs = [
            re.search(r"success_([0-9\.\-]+)", x).group(1)
            for x in checkpoint_flags
        ]
        int_epochs: Any = []
        for pieces in found_epochs:
            int_epochs.append([int(float(pieces)), '0'])

        last_epoch = sorted(int_epochs, reverse=True)[0]
        if last_epoch[1] == '0':
            epoch_to_load = str(last_epoch[0])
        else:
            epoch_to_load = '{0}.{1}'.format(last_epoch[0], last_epoch[1])

        model_path = os.path.join(ckpt_dir,
                                  "model_state_epoch_{}.th".format(epoch_to_load))
        training_state_path = os.path.join(ckpt_dir,
                                           "middle_training_state_{}".format(epoch_to_load))
        return (model_path, training_state_path)

    def find_latest_checkpoint(self) -> Tuple[str, str]:
        return self.find_latest_checkpoint_from_dir(self._serialization_dir)

    def restore_checkpoint(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass

    def best_model_state(self) -> Dict[str, Any]:
        """
        load最优的model参数
        """
        if self._serialization_dir:
            logger.info("loading best weights")
            best_model_state_path = os.path.join(
                self._serialization_dir, 'best.th')
            return torch.load(best_model_state_path)
        else:
            logger.info("cannot load best weights without `serialization_dir`, "
                        "so you're just getting the last weights")
            return {}
