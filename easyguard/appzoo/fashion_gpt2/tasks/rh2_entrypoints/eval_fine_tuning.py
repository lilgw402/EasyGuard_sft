import subprocess
import sys
import traceback
import logging

from rh2.sdk.env import get_rh2_env

rh2_env = get_rh2_env()


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


skip_fine_tuning = rh2_env.params.skip_fine_tuning
if skip_fine_tuning == 'true':
    print('skip_fine_tuning')
    exit(0)

model = rh2_env.inputs.get('model')
model_ckpt = get_only_file_hdfs_path(model.meta.hdfs_dir) if model else rh2_env.params.get('model_ckpt_fine_tuning')
if not model_ckpt:
    logging.error(f'No ckpt found under: {model.meta.hdfs_dir}')
    sys.exit(1)
print(f"model_ckpt: {model_ckpt}")

launch_args = rh2_env.params.launch_args
model_def_fine_tuning = rh2_env.params.model_def_fine_tuning


cmd = f'cd /opt/tiger/mariana && /bin/bash /opt/tiger/mariana/launch.sh tasks/gpt2/finetune/tnews/model.py --model={model_def_fine_tuning} --model.partial_pretrain={model_ckpt} {launch_args}'
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