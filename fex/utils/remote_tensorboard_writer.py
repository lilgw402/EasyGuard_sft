# -*- coding: utf-8 -*-
'''
Created on Nov-13-20 17:08
remote_tensorboard_writer.py
@author: liuzhen.nlp
Description:
'''

import logging
import time
import threading
import atexit
import subprocess

from torch.utils.tensorboard import SummaryWriter

from fex.utils.stop_watch import StopWatch

logger = logging.getLogger(__name__)


class RemoteSummaryWriter(SummaryWriter):
    """
    可以将metrics写入到hdfs的Writer, 继承子SummaryWriter
    """

    def __init__(self, hdfs_path=None, remote_flush_secs=300, **kawrgs):
        super().__init__(**kawrgs)
        if hdfs_path is None:
            logging.warning("no hdfs has been set, act as standard summary writer.")
            return
        self.hdfs_path = hdfs_path
        self.start_sync = True
        self.remote_flush_secs = remote_flush_secs
        self.sync_worker = threading.Thread(target=self.loop_sync)
        self.sync_worker.start()
        self.hadoop_bin = 'HADOOP_ROOT_LOGGER=ERROR,console /opt/tiger/yarn_deploy/hadoop/bin/hdfs'
        # https://stackoverflow.com/questions/865115/how-do-i-correctly-clean-up-a-python-object
        atexit.register(self.close)

    def loop_sync(self) -> None:
        """
        异步来将本地的log上传到hdfs目录，该过程是通过启动一个线程来完成的
        """
        watch = StopWatch()
        while self.start_sync:
            time.sleep(1)
            elapsed = watch.elapsed()
            if int(elapsed) % self.remote_flush_secs == 0:
                pipe = subprocess.Popen(
                    "{} dfs -copyFromLocal -f {}/* {}/".format(self.hadoop_bin, self.log_dir, self.hdfs_path), shell=True)
                pipe.wait()
        logger.info("summary writer remote sync is stoped.")

    def close(self) -> None:
        """
        Writer任务结束后，将线程和writer关闭
        """
        super().close()
        if not self.start_sync:
            return
        self.start_sync = False
        self.sync_worker.join()
        logger.info("stop summary writer.")
