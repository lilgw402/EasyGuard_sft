import subprocess
import sys
import traceback

from rh2.sdk.env import get_rh2_env

rh2_env = get_rh2_env()

model_ckpt = rh2_env.params.model_ckpt_zero_shot
dataset_index_list = rh2_env.params.dataset_index_list
model_def_zero_shot = rh2_env.params.model_def_zero_shot
model_in_nas = rh2_env.params.get('model_in_nas', False)

for i in dataset_index_list.split(','):
    cmd = f'/bin/bash /opt/tiger/mariana/tasks/gpt2/zero_shot_eval/scripts/run_new_ckpts.sh {model_def_zero_shot} {model_ckpt} {i}'
    if model_in_nas:
        cmd = f'/bin/bash /opt/tiger/mariana/tasks/gpt2/zero_shot_eval/scripts/run_sota_zero3_mnt.sh {model_def_zero_shot} {model_ckpt} {i}'
    elif '.yaml' not in model_def_zero_shot:
        cmd = f'/bin/bash /opt/tiger/mariana/tasks/gpt2/zero_shot_eval/scripts/run_sota.sh {model_def_zero_shot} {model_ckpt} {i}'

    print(f'run cmd: {cmd}', flush=True)
    try:
        sub_exit_code = subprocess.call(cmd, shell=True)
    except subprocess.CalledProcessError as ex:
        sub_exit_code = ex.returncode
    except Exception as e:
        sub_exit_code = 1
        traceback.print_exc()

    if sub_exit_code:
        sys.exit(sub_exit_code)