
import os
import sys
import random
import time

from fex.engine.launch import main
from fex import _logger as log


def cli():

    if 'ARNOLD_WORKER_GPU' in os.environ:
        gpu_num = str(int(os.environ['ARNOLD_WORKER_GPU']))
        log.info(f'Launching with [{gpu_num}] gpus found in `ARNOLD_WORKER_GPU`')
    else:
        # 如果没有 ARNOLD_WORKER_GPU，则默认是单机调试的情况。
        # 先根据 CUDA_VISIBLE_DEVICES 来确定gpu数，如果没设置，则检查机器有多少卡
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            gpu_num = str(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
            log.info(f'Launching with [{gpu_num}] gpus found in `CUDA_VISIBLE_DEVICES`')
        else:
            gpu_num = os.popen('lspci | grep -i NVIDIA | wc -l').read().strip()
            log.info(f'Launching with [{gpu_num}] gpus found from `lspci | grep -i NVIDIA | wc -l`')
        os.environ['ARNOLD_WORKER_NUM'] = '1'
        os.environ['ARNOLD_WORKER_GPU'] = gpu_num
        os.environ['METIS_TASK_INDEX'] = '0'
        os.environ['METIS_WORKER_0_HOST'] = '0.0.0.0'
        os.environ['METIS_WORKER_0_PORT'] = str(random.randint(1, 9) * 1000 + random.randint(111, 999))

    os.environ['PYTHONPATH'] = f"{os.environ.get('PYTHONPATH', '')}:."  # cur path

    launch_args = []
    launch_args.extend(['--nproc_per_node', os.environ['ARNOLD_WORKER_GPU']])
    launch_args.extend(['--nnodes', os.environ['ARNOLD_WORKER_NUM']])
    launch_args.extend(['--node_rank', os.environ['METIS_TASK_INDEX']])
    launch_args.extend(['--master_addr', os.environ['METIS_WORKER_0_HOST']])
    launch_args.extend(['--master_port', os.environ['METIS_WORKER_0_PORT']])
    sys.argv = sys.argv[:1] + launch_args + sys.argv[1:]
    # print argvs
    log.info('args:')
    for arg in sys.argv:
        log.info(arg)
    time.sleep(2)  # give you one moment to read log

    main()
