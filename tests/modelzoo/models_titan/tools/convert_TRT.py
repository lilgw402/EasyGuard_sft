"""
Simple test for model conversion torch -> onnx -> trt.
"""
import os
import logging
import time
from numpy import mean
from functools import wraps
import torch
import argparse
from shutil import copy2

from anyon.utils import proto_utils, hdfs_utils, path_utils, misc
from anyon.proto import anyon_core_pb2
from anyon.mlsys import bridge
from anyon import mlsys as mlsys_utils
from anyon.details.benchmark.benchmark_qs_v2_engine import parse_benchmark_file
from utils import get_sub_graph, onnx_graph_to_trt, torch_to_onnx

import titan
from titan.utils.hdfs import (
    hdfs_put,
    has_hdfs_path_prefix
)
from titan.utils.misc import has_http_path_prefix


def time_costing(func):
    @wraps(func)
    def core(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        time_cost = (end_time - start_time) / 60
        return res, time_cost
    return core


@time_costing
def test_latency(module_path: str,
                 batch_size: int,
                 output_dir: str,
                 repeat: int = 20,
                 warmup: int = 5,
                 trt_precision: str = 'FP16'):
    """Convert onnx model to trt & test trt model latency."""
    tmp_output_dir = os.path.abspath('./qps_res')
    path_utils.create_path_if_not_exists(tmp_output_dir)

    # step1: get anyon_model_graph
    onnx_sub_graph = get_sub_graph(path=module_path, graph_type='onnx')
    trt_sub_graph = onnx_graph_to_trt(onnx_sub_graph,
                                      max_batch_size=batch_size,
                                      trt_precision=trt_precision,
                                      output_dir=tmp_output_dir)
    anyon_model_graph = proto_utils.dict_to_proto(
        {'sub_graphs': [trt_sub_graph]}, anyon_core_pb2.Graph)

    # step2: dump qs_info to qs deploy file
    qs_inf = bridge.convert_anyon_graph_to_inference_param(
        anyon_model_graph, max_batch_size=[batch_size])
    deploy_file = os.path.join(tmp_output_dir, 'qs_deploy.prototxt')
    misc.dump_proto_to_text_file(deploy_file, qs_inf)

    # step3: test qps from qs deploy file
    min_latency = []
    max_qps = []
    for i in range(warmup + repeat):
        benchmark_file = os.path.join(tmp_output_dir,
                                      'benchmark_qs_deploy_{}.log'.format(i))
        mlsys_utils.test_qps(deploy_file,
                             dump_filename=benchmark_file,
                             use_v2=True)
        # parse the benmark file to get structured info
        info = parse_benchmark_file(benchmark_file)
        if i >= warmup:
            min_latency.append(info['min_latency'])
            max_qps.append(info['max_qps'])

    mean_latency = mean(min_latency)
    mean_qps = mean(max_qps)

    # step4: save output
    if has_hdfs_path_prefix(output_dir):
        hdfs_put(tmp_output_dir, output_dir, overwrite=True)
    elif has_http_path_prefix(output_dir):
        raise TypeError(f'http path:{output_dir} is not supported.')
    else:  # local path
        if output_dir[-1] == '/':
            output_dir = output_dir[:-1]
        dirname = os.path.dirname(output_dir)
        if dirname != '' and not os.path.exists(dirname):
            os.makedirs(dirname)
        os.system(f'cp -r {tmp_output_dir} {output_dir}')

    return mean_latency, mean_qps


def parse_args():
    parser = argparse.ArgumentParser(description='Titan Model Conversion (Torch -> ONNX -> TRT)')
    parser.add_argument('--model_name', type=str, required=True, help='model name')
    parser.add_argument('--model_output_dir', type=str, required=True,
                        help='model output directory on hdfs, http or local')
    parser.add_argument('--tos_bucket', type=str, help='model zoo bucket name')
    parser.add_argument('--tos_access_key', type=str,
                        help='model zoo bucket access key')
    parser.add_argument('--tos_idc', type=str, default='',
                        help='model zoo bucket idc, for sg bucket, please specify sgcomm1')
    parser.add_argument('--model_version', type=str, help='model version')
    parser.add_argument('--model_backend', type=str, default='titan',
                        help='model backend, currently support titan or timm models')
    parser.add_argument('--model_path', type=str, help='model path on hdfs, http or local')
    parser.add_argument('--input_shape', type=str, default='3,224,224',
                        help='model input image shape, string split by comma. Default is 3,224,224')
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='Number of classes, default is 1000.')
    parser.add_argument('--pretrained', action='store_true',
                        help='whether to use pretrained weights, default is False.')
    parser.add_argument('--features_only', action='store_true',
                        help='whether to only output features from the model, default is False.')
    parser.add_argument("--verify", action='store_true',
                        help="whether to verify numerical difference between original model and converted model.")
    parser.add_argument("--batch_size", default="1", help="batch size")
    return parser.parse_args()


def main(args):
    # create model
    tos_helper = None
    if args.tos_bucket and args.tos_access_key:
        # init tos helper
        tos_helper = titan.TOSHelper(
            args.tos_bucket,
            args.tos_access_key,
            idc=args.tos_idc)

    model = titan.create_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        pretrained_version=args.model_version,
        pretrained_uri=args.model_path,
        backend=args.model_backend,
        features_only=args.features_only,
        num_classes=args.num_classes,
        tos_helper=tos_helper)
    dummy_input = torch.randn(eval("1," + args.input_shape))
    model.eval()

    print('Starting ONNX conversion')
    input_name = ['input']
    output_name = ['output']
    tmp_output_path = './tmp/' + args.model_name + '.onnx'
    if not os.path.exists('./tmp'):
        os.mkdir('./tmp')
    torch.onnx.export(
        model,
        dummy_input,
        tmp_output_path,
        verbose=True,
        input_names=input_name,
        output_names=output_name,
        opset_version=13,
        dynamic_axes={'input':{0:'batch_size'}, 'output':{0:'batch_size'},})

    str_batch_size = args.batch_size
    (latency, qps), time_cost = test_latency(module_path=tmp_output_path,
                                             batch_size=int(str_batch_size),
                                             output_dir=args.model_output_dir)
    logging.info('Latency: {:.3f}ms, qps: {}, time_cost: {:.3f}s'.format(
        latency, qps, time_cost))


if __name__ == '__main__':
    args = parse_args()

    main(args)
