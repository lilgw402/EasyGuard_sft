
import argparse
import os
from shutil import copy2
from titan.utils.hdfs import (
    _hdfs_get,
    has_hdfs_path_prefix
)
from titan.utils.misc import (
    check_tos_model_path,
    has_http_path_prefix,
    download_http_to_local_file
)
from titan.utils.tos_helper import TOSHelper

LOCAL_WORK_DIR = "./work_dir"


def parse_args():
    parser = argparse.ArgumentParser(description='Download dataset from hdfs')
    # General
    parser.add_argument('--tos_bucket', type=str,
                        help='model zoo bucket name')
    parser.add_argument('--tos_access_key', type=str,
                        help='model zoo bucket access key')
    parser.add_argument('--tos_idc', type=str, default='',
                        help='model zoo bucket idc, for sg bucket, please specify sgcomm1')
    parser.add_argument('--src_model_path', type=str,
                        help='source model path, can be either from hdfs, http url or local file')
    parser.add_argument('--dst_model_path', type=str,
                        help='output path on model zoo tos bucket. Must be organized \
                        in the format: \"backend/model_name/model_version/model.pth\"')

    return parser.parse_args()


def main():
    args = parse_args()

    local_work_dir = LOCAL_WORK_DIR
    if not os.path.exists(local_work_dir):
        os.makedirs(local_work_dir)

    model_name = os.path.basename(args.src_model_path)
    if has_hdfs_path_prefix(args.src_model_path):
        _hdfs_get(
            args.src_model_path,
            os.path.join(local_work_dir, model_name),
            overwrite=True)
    elif has_http_path_prefix(args.src_model_path):
        download_http_to_local_file(
            args.src_model_path,
            os.path.join(local_work_dir, model_name),
        )
    elif os.path.exists(args.src_model_path):
        if not os.path.samefile(args.src_model_path, os.path.join(local_work_dir, model_name)):
            copy2(args.src_model_path, os.path.join(local_work_dir, model_name))
    else:
        raise FileNotFoundError(
            f'File path {args.src_model_path} does not exist.')

    # validate the model path saved on tos
    # valid path: {backend: haggs/timm}/{model_name, i.e., resnet50}/{model_version}/{file_name}
    if not check_tos_model_path(args.dst_model_path):
        raise ValueError(f'TOS model path {args.dst_model_path} is invalid.')

    # if upload model to sg bucket, need to specify idc='sgcomm1'.
    tos_helper = TOSHelper(
        args.tos_bucket,
        args.tos_access_key,
        idc=args.tos_idc)

    tos_helper.upload_model_to_tos(
        input_path=os.path.join(local_work_dir, model_name),
        filename=args.dst_model_path,
    )


if __name__ == '__main__':
    main()
