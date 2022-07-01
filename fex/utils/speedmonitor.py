# -*- coding: utf-8 -*-
'''
Created on Nov-16-20 16:53
speedometer.py
@author: liuzhen.nlp
Description:
'''
from typing import Dict, Any

import time
import logging
from logging import Logger

logger = logging.getLogger(__name__)


class Speedmonitor:
    """
    用来监控整个训练中的训练速度
    """

    def __init__(self, batch_size: int, logger: Logger, frequent: int = 50,
                 batches_per_epoch: int = None, total_epochs: int = None, rank: int = 0, gradient_accumulate_steps: int = 1):
        self.logger = logger
        self.batch_size = batch_size
        self.frequent = frequent
        self.batches_per_epoch = batches_per_epoch
        self.total_epochs = total_epochs
        self.epoch = -1
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.data_iter_elapsed = 0.0
        self.train_step_start_elapsed = 0.0
        self.forward_elapsed = 0.0
        self.backward_step_elapsed = 0.0
        self.optimizer_step_elapsed = 0.0
        self.total_loss = 0.0
        self.current_loss = 0.0
        self.rank = rank
        self.gradient_accumulate_steps = gradient_accumulate_steps

    def __call__(self, current_epoch, current_batch, param, extra_metric_dic):
        """Callback to Show speed."""
        count = current_batch
        if self.last_count > count:
            self.init = False
        self.last_count = count
        self.data_iter_elapsed += param.data_iter
        self.train_step_start_elapsed += param.train_step_start
        self.forward_elapsed += param.forward
        self.backward_step_elapsed += param.backward_step
        self.optimizer_step_elapsed += param.optimizer_step
        self.total_loss += param.loss
        self.current_loss = param.loss * self.gradient_accumulate_steps

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                data_iter_elapsed = self.data_iter_elapsed / self.frequent
                train_step_start_elapsed = self.train_step_start_elapsed / self.frequent
                forward_elapsed = self.forward_elapsed / self.frequent
                backward_step_elapsed = self.backward_step_elapsed / self.frequent
                optimizer_step_elapsed = self.optimizer_step_elapsed / self.frequent
                avg_loss = (self.total_loss / self.frequent) * self.gradient_accumulate_steps

                eta = ((self.total_epochs - current_epoch - 1) * self.batches_per_epoch +
                       self.batches_per_epoch - current_batch) * self.batch_size / speed
                eta = int(eta / 60.0)
                eta_m = eta % 60
                eta_h = int((eta - eta_m) / 60) % 24
                eta_d = int((eta - eta_m - eta_h * 60) / (24 * 60))
                s = ''

                prefix = "Epoch[%d] Batch [%d]\t" % (current_epoch, count)
                s = prefix + "Speed: %.2f samples/s ETA: %d d %2d h %2d m \tData: %.3f Iter: %.3f F: %.3f B: %.3f O: %.3f\tLoss: %.4f\tAvg_Loss: %.4f" % (
                    speed, eta_d, eta_h, eta_m, data_iter_elapsed, train_step_start_elapsed, forward_elapsed, backward_step_elapsed, optimizer_step_elapsed, self.current_loss, avg_loss)
                for k, v in extra_metric_dic.items():
                    s += '\t%s: %.4f' % (k, v)

                if self.rank is not None:
                    s = 'Rank[%3d]' % self.rank + s

                self.logger.info(s)
                self.tic = time.time()
                self.data_iter_elapsed = 0.0
                self.train_step_start_elapsed = 0.0
                self.forward_elapsed = 0.0
                self.backward_step_elapsed = 0.0
                self.optimizer_step_elapsed = 0.0
                self.train_step_end_elapsed = 0.0
                self.total_loss = 0.0
                self.current_loss = 0.0
        else:
            self.init = True
            self.epoch += 1
            s = "Epoch[%d] Batch [%d]\tSpeed: - samples/sec ETA: - d - h - m" % (
                current_epoch, 0)

            if self.rank is not None:
                s = 'Rank[%3d]' % self.rank + s

            self.logger.info(s)
            self.tic = time.time()

    def state_dict(self) -> Dict[str, Any]:
        """
        A ``Trainer`` can use this to serialize the state of the metric tracker.
        """
        return {
            "batch_size": self.batch_size,
            "frequent": self.frequent,
            "batches_per_epoch": self.batches_per_epoch,
            "total_epochs": self.total_epochs,
            "epoch": self.epoch,
            "gradient_accumulate_steps": self.gradient_accumulate_steps
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        A ``Trainer`` can use this to hydrate a metric tracker from a serialized state.
        """
        self.batch_size = state_dict["batch_size"]
        self.frequent = state_dict["frequent"]
        self.batches_per_epoch = state_dict["batches_per_epoch"]
        self.total_epochs = state_dict["total_epochs"]
        self.epoch = state_dict["epoch"]
        self.gradient_accumulate_steps = state_dict["gradient_accumulate_steps"]
