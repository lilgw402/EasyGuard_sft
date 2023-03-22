import subprocess
import sys
import traceback

from cruise.utilities.hdfs_io import hdfs_open, hexists, hcopy

from bytedrh2 import job
from rh2.sdk.env import get_rh2_env
import logging

rh2_env = get_rh2_env()
job_run = job.get_current_run()

def get_only_file_hdfs_path(hdfs_dir):
    try:
        # Get the HDFS path of the only file in the directory
        output = subprocess.check_output(['hdfs', 'dfs', '-ls', hdfs_dir]).decode('utf-8').strip()
        lines = output.split('\n')
        if len(lines) == 2:  # Only the header and the file
            file_path = lines[1].split()[-1]
            return file_path
        else:
            print(f"Error: There is not exactly one file in the directory {hdfs_dir}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None

def save_ckpt(model_ckpt):
    print(f'update all input ckpt to {model_ckpt}')
    if 'model_ckpt_zero_shot' in rh2_env.params:
        job_run.update_param('model_ckpt_zero_shot', model_ckpt)
    if 'model_ckpt_fine_tuning' in rh2_env.params:
        job_run.update_param('model_ckpt_fine_tuning', model_ckpt)
    if 'output_model' in rh2_env.outputs:
        hcopy(model_ckpt, output_model.meta.hdfs_dir)


merge_zero3_ckpt = rh2_env.params.merge_zero3_ckpt
input_model = rh2_env.inputs.get('input_model')
model_ckpt = get_only_file_hdfs_path(input_model.meta.hdfs_dir) if input_model else rh2_env.params.get('model_ckpt')
output_model = rh2_env.outputs.get('output_model')


if merge_zero3_ckpt != 'true':
    print('skip merge_zero3_ckpt')

    if model_ckpt != '':
        save_ckpt(model_ckpt)

    sys.exit(0)

# example: hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/models/gpt/13b_compactdata_v220_20220210/checkpoints/global_step_20000
# hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/models/gpt/13b_compactdata_v220_20220210/checkpoints/global_step_20000/zero3_merge_states.pt

new_model_ckpt = f'{model_ckpt}/zero3_merge_states.pt'
save_ckpt(new_model_ckpt)

if hexists(new_model_ckpt):
    print(f'new_model_ckpt={new_model_ckpt} exists, skip')
    sys.exit(0)

cmd = f'cd /opt/tiger/mariana/tasks/gpt2 && python3 merge_zero3_ckpt.py --dtype=bf16 --checkpoint_dir={model_ckpt}'
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