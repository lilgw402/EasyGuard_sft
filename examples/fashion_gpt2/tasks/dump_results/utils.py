import subprocess
import logging
from typing import List
import os
import tensorflow as tf

LOCAL_PATH = 'temp.json'

def read_from_hdfs(hdfs_path):
    print("===Getting from hdfs: {}===".format(hdfs_path))

    if os.path.exists(LOCAL_PATH):
        os.remove(LOCAL_PATH)

    cmd = ["hdfs", "dfs", "-get", hdfs_path, LOCAL_PATH]

    try:
        subprocess.check_output(cmd)
        with open(LOCAL_PATH, 'r') as f:
            return f.read()
    except Exception as e:
        logging.error(e)
        return None

def download_from_hdfs(hdfs_path, local_dir) -> None:
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)

    print("Downloading from: {}".format(hdfs_path))

    result = subprocess.run(["hadoop", "fs", "-get", hdfs_path, local_dir], capture_output=True)
    if result.returncode == 0:
        print("File downloaded successfully: ".format(hdfs_path))
    else:
        print("Error downloading {}: {}".format(hdfs_path, result.stderr.decode().strip()))


def list_hdfs_dir(hdfs_dir: str) -> List[str]:
    cmd = ['hdfs', 'dfs', '-ls', hdfs_dir]
    output = subprocess.check_output(cmd).decode().strip()
    file_lines = [line for line in output.split('\n')]
    file_paths = [line.split()[-1] for line in file_lines]
    return file_paths


def get_loss_from_tfevent_file(tfevent_filename):
    """
    :param tfevent_filename: the name of one tfevent file
    :return: loss (list)
    """
    loss_val_list = []
    for event in tf.compat.v1.train.summary_iterator(tfevent_filename):
        for value in event.summary.value:
            if value.HasField('simple_value'):
                if value.tag == "training/loss":
                    loss_val_list.append((event.step, value.simple_value))
    return loss_val_list
    
def get_target_loss(file_path, target_step):
    loss_val_list = get_loss_from_tfevent_file(file_path)
    min_step_distance = target_step
    result_loss = -1
    result_step = 0

    for step, loss in loss_val_list:
        distance = abs(target_step - step)
        if distance < min_step_distance:
            min_step_distance = distance
            result_loss = loss
            result_step = step
    return result_loss, result_step