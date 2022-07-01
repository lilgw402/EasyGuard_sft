import argparse
import logging
import time
import os

import psutil


def check_dangling_process(do_kill):
    proc_cnt = 0
    self_pid = os.getpid()
    for proc in psutil.process_iter():
        try:
            pid = proc.pid
            ppid = proc.ppid()
            cmd = proc.cmdline()
            if ppid == 1 and pid != self_pid:
                if cmd and 'python3' in cmd[0] and 'torch.distributed.run' not in cmd:
                    proc_cnt += 1
                    if do_kill:
                        logging.info(f'[watchdog] Killing dangling python process {pid} cmd {cmd}')
                        proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return proc_cnt


def is_training():
    for proc in psutil.process_iter():
        try:
            cmd = proc.cmdline()
            if 'torch.distributed.run' in cmd:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-delay', type=int, default=300)
    parser.add_argument('--check-interval', type=int, default=1)
    parser.add_argument('--kill-delay', type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        datefmt=r'%Y/%m/%d %H:%M:%S',
        format=r'[%(asctime)s %(filename)s#%(lineno)3d] %(levelname)s: %(message)s'
    )

    logging.info(f'[watchdog] Sleeping for {args.start_delay}s before starting ...')
    time.sleep(args.start_delay)
    logging.info('[watchdog] Starting elastic watchdog ...')

    while is_training():
        cnt = check_dangling_process(do_kill=False)
        if cnt > 0:
            time.sleep(args.kill_delay)
            check_dangling_process(do_kill=True)
        time.sleep(args.check_interval)

    logging.info('[watchdog] Watchdog exited.')


if __name__ == '__main__':
    main()
