"""
Simple test for model conversion torch -> TS/ONNX
"""
import torch 
import argparse
import onnxruntime as rt
import numpy as np
import os
from shutil import copy2

import titan
from titan.utils.hdfs import (
    hdfs_put,
    has_hdfs_path_prefix
)
from titan.utils.misc import has_http_path_prefix


def parse_args():
    parser = argparse.ArgumentParser(description='Titan Model Conversion (Torch -> ONNX or TorchScript)')
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
    return parser.parse_args()


def save_model(tmp_path, final_path):
    if has_hdfs_path_prefix(final_path):
        hdfs_put(tmp_path, final_path, overwrite=True)
    elif has_http_path_prefix(final_path):
        raise TypeError(f'http path:{final_path} is not supported.')
    else:  # local path
        if not os.path.exists(os.path.dirname(final_path)):
            os.makedirs(os.path.dirname(final_path))
        copy2(tmp_path, final_path)


def onnx_converter(args, model, dummy_input):
    print('Starting ONNX conversion')
    num_classes = model.num_classes
    input_name = ['input']
    output_name = ['output']
    final_output_path = os.path.join(args.model_output_dir, args.model_name + '.onnx')
    tmp_output_path = './tmp/'+ args.model_name + '.onnx'
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
        dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'},})
    save_model(tmp_output_path, final_output_path)
    print(f'Successfully exported ONNX model: {final_output_path}')
    
    if args.verify:
        # check by onnx
        import onnx
        onnx_model = onnx.load(tmp_output_path)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        with torch.no_grad():
            pytorch_result = model(dummy_input)
            pytorch_result = pytorch_result.cpu().detach().numpy()
        print(pytorch_result.shape)

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession(tmp_output_path)
        onnx_result = sess.run(
            None, {net_feed_input[0]: dummy_input.cpu().detach().numpy()})[0]
        print(onnx_result.shape)

        # compare results
        np.testing.assert_allclose(
            pytorch_result.astype(np.float32) / num_classes,
            onnx_result.astype(np.float32) / num_classes,
            rtol=1e-5,
            atol=1e-5,
            err_msg='The outputs are different between Pytorch and ONNX')
        print('The outputs are same between Pytorch and ONNX')


def torchscript_converter(args, model, dummy_input):
    print('Starting Torchscript conversion')
    num_classes = model.num_classes
    r18_traced = torch.jit.trace(model, dummy_input)
    final_output_path = os.path.join(args.model_output_dir, args.model_name + '.pt')
    tmp_output_path = './tmp/' + args.model_name + '.pt'
    if not os.path.exists('./tmp'):
        os.mkdir('./tmp')
    r18_traced.cuda()
    r18_traced.save(tmp_output_path)
    save_model(tmp_output_path, final_output_path)
    print(f'Successfully exported TorchScript model: {final_output_path}')

    if args.verify:
        # check by TS
        loaded = torch.jit.load(final_output_path)

        # check the numerical value
        # get pytorch output
        with torch.no_grad():
            pytorch_result = model(dummy_input)
            pytorch_result = pytorch_result.cpu().detach().numpy()
        print(pytorch_result.shape)

        # get TS output
        out_TS = loaded(dummy_input)
        out_TS = out_TS.cpu().detach().numpy()

        # compare results
        np.testing.assert_allclose(
            pytorch_result.astype(np.float32) / num_classes,
            out_TS.astype(np.float32) / num_classes,
            rtol=1e-5,
            atol=1e-5,
            err_msg='The outputs are different between Pytorch and TorchScript')
        print('The outputs are same between Pytorch and TorchScript')


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

    dummy_input = torch.randn(eval("1," + args.input_shape)).cuda()
    model.cuda()
    model.eval()
    print("Is the model in training mode? " + str(model.training))

    # convert to onnx model
    onnx_converter(args, model, dummy_input)

    # convert to torchscript model
    torchscript_converter(args, model, dummy_input)


if __name__ == '__main__':
    args = parse_args()

    main(args)
