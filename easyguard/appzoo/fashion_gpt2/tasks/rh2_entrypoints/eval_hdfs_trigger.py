"""

Trigger eval pipeline run base on HDFS scanning result 

- read index.json from /user/eval_pipeline/index/*
- foreach ckpt path, find target ckpt
- foreach ckpt, check hdfs if eval pipeline launched
- if not, launch and write record to hdfs
- write all target ckpt result (path, pipeline id, new launch) to tracking table

"""

import argparse
import json
import requests
import time
import os
import subprocess

import wandb
import pandas as pd

from datetime import datetime
from typing import IO, Any, List
from cruise.utilities.hdfs_io import hdfs_open

INDEX_JSON_PATH = 'hdfs://haruna/home/byte_aml_platform/user/eval_pipeline/index/index.json'
RESULT_JSON_PATH = 'hdfs://haruna/home/byte_aml_platform/user/eval_pipeline/result/result.json'

class HttpClient(object):

    def __init__(self, name, timeout=3):
        self.name = name
        self.timeout = timeout

    def do(self, method, url, ctx=None, **kwargs):
        # TODO do hook here
        # emtrics
        # excpetion handling, retry
        ctx = ctx or {}
        start_t = time.time() * 1000
        try:
            if 'timeout' not in kwargs:
                kwargs['timeout'] = self.timeout
            resp = requests.request(method, url, **kwargs)
            latency = time.time() * 1000 - start_t
            return resp
        except Exception as ex:
            latency = time.time() * 1000 - start_t
            raise Exception('[HttpClient] err calling service %s, err: %s', self.name, ex)

def get_rh2_token(username: str, token: str) -> str:
    api = "https://rh2.bytedance.net/api/v1/user/get_tenant_access_token"
    body = {
        "user_name": username,
        "access_token": token
    }
    header = {
        "Content-Type": "application/json",
    }
    response = client.do('post', api, json=body, headers=header, auth=None)
    return response.json()['tenantAccessToken']

def trigger(trigger_id: str, params: list, username: str, token: str) -> str:
    api = 'https://rh2.bytedance.net/api/v1/trigger/api_trigger'
    data = {
        "pipeline_params": {
            "params": params
        }, 
        'trigger_id': trigger_id
    }
    header = {
        "Content-Type":     "application/json",
        "RH2-TENANT-TOKEN": get_rh2_token(username, token),
        "RH2-USERNAME":     username,
        "Rhcli-Username":   username,
    }
    
    response = client.do('post', api, json=data, headers=header, auth=None)
    res = response.json()
    
    print('trigger rh2 pipeline')
    print(data)
    print(res)
    return res['runId']

client = HttpClient("merlin", timeout=20)

def hget(path: str) -> str:
    with hdfs_open(path, 'r') as f:
        return f.read()

def list_hdfs_dir_with_date(hdfs_dir: str) -> List:
    cmd = ['hdfs', 'dfs', '-ls', hdfs_dir]
    output = subprocess.check_output(cmd).decode().strip()
    file_lines = [line for line in output.split('\n') if len(line.split()) > 5]
    file_path_and_date = [(line.split()[-1], line.split()[-3]) for line in file_lines]
    return file_path_and_date


def find_and_filter_ckpts(
    ckpt_root_path: str, ckpt_name: str, step_reminder: int, last_n_days: int) -> List:
    # find global_step_250000
    # return global_step_250000/{ckpt_name}, step

    ckpts = []
    for path, datestr in list_hdfs_dir_with_date(ckpt_root_path):
        if 'global_step' not in path:
            continue

        step = int(path.split('/')[-1].replace('global_step_', ''))
        if step > 0 and step % step_reminder != 0:
            continue

        d = datetime.strptime(datestr, "%Y-%m-%d")
        if (datetime.today() - d).days > last_n_days:
            continue

        if ckpt_name:
            ckpts.append((f'{path}/{ckpt_name}', step))
        else:
            ckpts.append((f'{path}', step))

    return ckpts


def get_json_from_hdfs(path: str):
    return json.loads(hget(path))


def trigger_eval_pipeline(
    name: str, step: int, ckpt_path: str, 
    params: dict, trigger_id: str,
    username: str, token: str):

    params_dict = params.copy()
    params_dict['model_ckpt'] = ckpt_path
    params_dict['model_ckpt_zero_shot'] = ''
    params_dict['model_ckpt_fine_tuning'] = ''
    params_dict['project_name'] = f'{name}_step{step}'

    pipeline_run_id = trigger(trigger_id, [
        {
            "name": k,
            "value": params_dict[k]
        }
        for k in params_dict
    ], username, token)

    return {
        'name': name,
        'ckpt_path': ckpt_path,
        'trigger_id': trigger_id,
        'pipeline_run_id': pipeline_run_id,
        'new_launch': 'true',
        "params": str(params)
    }

def write_result_map_to_hdfs(result_map: dict, path: str):
    content = json.dumps(result_map)
    with hdfs_open(path, 'w') as f:
        f.write(content.encode())

def get_table_data(obj):
    df = None
    if type(obj) == list:
        df = pd.DataFrame(obj)
    else:
        df = pd.DataFrame([obj[k] for k in obj])
    
    if 'ckpt_path' in df.columns:
        df = df.sort_values('ckpt_path')
    
    records = df.to_records()
    
    return [list(r)[1:] for r in records], list(df.columns)

def write_tracking(
    tracking_project: str, 
    trigger_map: dict, 
    index_list: List, 
    pending_list: List,
    msg: str):

    print(f"Dumping results to tracking: {tracking_project}, msg: {msg}")
    
    wandb.init(project=tracking_project, name=time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()))

    data, columns = get_table_data(index_list)
    table = wandb.Table(data=data, columns=columns)
    wandb.log({"index": table})

    table = wandb.Table(data=[
        [item['name'], item['ckpt_path']]
        for item in pending_list
    ], columns=['name', 'ckpt_path'])
    wandb.log({"pending": table})

    wandb.summary['msg'] = msg

    wandb.finish()

def main(args):
    index_list = get_json_from_hdfs(INDEX_JSON_PATH)
    print('index_list', index_list)
            
    result_map = get_json_from_hdfs(RESULT_JSON_PATH)
    print('result_map', result_map)

    ckpt_list = []
    for item in index_list:
        if item.get('enabled') == False:
            print(f'skip item={item["prober_config"]["name"]}')
            continue
        
        prober_config = item['prober_config']
        step_reminder = prober_config.get('step_reminder', args.step_reminder)
        last_n_days = prober_config.get('last_n_days', args.last_n_days)

        ckpts = find_and_filter_ckpts(
            prober_config['scan_root_path'], 
            prober_config['ckpt_name'], 
            step_reminder,
            last_n_days)
        ckpt_list.extend([(item, c[0], c[1]) for c in ckpts])

    pending_list = []
    new_trigger_cnt = 0
    for item, ckpt_path, step in ckpt_list:
        if ckpt_path in result_map:
            print(f'skip: {ckpt_path}')
            result_map[ckpt_path]['new_launch'] = 'false'
            continue
        
        prober_config = item['prober_config']

        print(f'find new ckpt for {prober_config["name"]}: {ckpt_path}')

        if new_trigger_cnt == args.trigger_limit:
            pending_list.append({
                'name': prober_config["name"], 
                'ckpt_path': ckpt_path
            })
            print('reach trigger_limit, skip')
            continue

        info = trigger_eval_pipeline(
            prober_config["name"],
            step,
            ckpt_path,
            item["trigger_params"],
            prober_config['trigger_id'],
            args.username, 
            args.token)
        
        result_map[ckpt_path] = info
        new_trigger_cnt += 1

    write_result_map_to_hdfs(result_map, RESULT_JSON_PATH)
    
    write_tracking(args.tracking_project, result_map, index_list, pending_list,
        f'all trigger={len(result_map)}, new trigger={new_trigger_cnt}, pending={len(pending_list)}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HDFS trigger')
    
    parser.add_argument('--last_n_days', type=int, default=0, help='探查几天内创建出的ckpt，比如1天内')
    parser.add_argument('--step_reminder', type=int, default=50000, help='step余数，满足取余为0再触发，比如50000')
    parser.add_argument('--trigger_limit', type=int, default=1, help='每次最大新触发的执行数')
    parser.add_argument('--tracking_project', type=str, default='eval_hdfs_trigger', help='输出探查日志的Tracking项目')
    parser.add_argument('--username', type=str, help='username')
    parser.add_argument('--token', type=str, help='token')

    args = parser.parse_args()
    print(args)
    
    main(args)

    print('done')
