import wandb
from rh2.sdk.env import get_rh2_env
import json
from utils import read_from_hdfs, get_target_loss, download_from_hdfs, list_hdfs_dir
import sys 
from typing import List
import re

from rh2.sdk.env import get_rh2_env
from bytedrh2.http_client import _get_rh2_client
import logging

rh2_env = get_rh2_env()

def _get_metrics(path) -> dict:
    content = read_from_hdfs(path)
    metrics = json.loads(content)
    return metrics
     
def _get_event_paths(hdfs_path: str) -> List[str]:
    main_dir = hdfs_path.split('/checkpoints/')[0]
    event_paths = list_hdfs_dir(main_dir)
    event_paths = [path for path in event_paths if 'events.out.tfevents.' in path]
    return event_paths

def _get_target_step(hdfs_path: str) -> int:
    match = re.search(r"global_step_(\d+)", hdfs_path)

    if match:
        number = int(match.group(1))
        print('Target step is: {}'.format(number))
        return number
    else:
        print("Cannot extract target step from hdfs path: {}".format(hdfs_path))
        return None

def _get_global_train_loss(ckpt_path):
    target_step = _get_target_step(ckpt_path)
    if not target_step:
        return -1, 0

    event_paths = _get_event_paths(ckpt_path)
    closest_step = 0
    global_loss = -1

    for path in event_paths:
        download_from_hdfs(path,'./events/')
        file_name = path.split("/")[-1]
        local_path = f"./events/{file_name}"
        curr_loss, curr_step = get_target_loss(local_path, target_step)
        print("Local loss: {}, step: {}, file: {}".format(curr_loss, curr_step, file_name))

        if abs(curr_step - target_step) < abs(closest_step - target_step):
            closest_step = curr_step
            global_loss = curr_loss
        
        elif abs(curr_step - target_step) == abs(closest_step - target_step) and abs(curr_loss) < abs(global_loss):
            closest_step = curr_step
            global_loss = curr_loss
            
        print("Global loss: {}, step: {}".format(global_loss, closest_step))

    return global_loss, closest_step


def _get_params() ->dict:
    client = _get_rh2_client()
    job_run_id = rh2_env.job_run_id
    _, jr_resp = client.get_job_run(job_run_id)
    param_names = rh2_env.params.get('params_to_track', 'model_ckpt_zero_shot, model_ckpt_fine_tuning, tracking_extra_note')
    param_names = list(map(lambda name: name.strip(), param_names.split(',')))

    pipeline_run_id = jr_resp.job_run.meta.pipeline_run_id or '4d2c71f87dd7c6b0'
    if not pipeline_run_id:
        return {}

    _, pr_resp = client.get_pipeline_run(pipeline_run_id)
    params = pr_resp.pipeline_run.meta.pipeline_run_params.params
    d = {}
    for param in params:
        if param.name in param_names:
            d[param.name] = param.value
    return d


def dump_results():
    """dump results into tracking, will skip if been dumpped before
    """
    
    rh2_env = get_rh2_env()
    project_id = rh2_env.params.tracking_project_id

    model_ckpt_zero_shot = rh2_env.params.model_ckpt_zero_shot
    pipeline_run_id = rh2_env.get('instance', {}).get('pipeline_run_id') or '4d2c71f87dd7c6b0' 

    host = rh2_env.host
    pipeline_url = 'https://{}/rh2/pipeline_run/details/{}'.format(host, pipeline_run_id)

    path = f'hdfs://haruna/home/byte_aml_platform/user/eval_pipeline/{pipeline_run_id}.json'
    metrics = _get_metrics(path)
    project_name = rh2_env.params.get('project_name') or pipeline_run_id

    print("Dumping results to tracking: {}".format(project_id))
    wandb.init(project=project_id, name=project_name)
    wandb.summary['pipeline_url'] = pipeline_url
    
    for k, v in metrics.items():
        value = v
        try:
            value = round(float(v), 6)
        except:
            pass
        wandb.summary[k] = value

    global_train_loss, step = _get_global_train_loss(model_ckpt_zero_shot)
    if global_train_loss:
        wandb.summary['global_train_loss'] = global_train_loss
        wandb.summary['step'] = step
    
    params = _get_params()
    print("Custom params: ", params)
    for k, v in params.items():
        wandb.summary[k] = v

    wandb.finish()

if __name__ == '__main__':
    dump_results()
