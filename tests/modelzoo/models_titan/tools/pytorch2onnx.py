import argparse
import numpy as np
import onnxruntime as rt
import torch
import torch._C
import torch.serialization
from titan import TOSHelper, create_model

torch.manual_seed(3)


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    segs = rng.randint(
        low=0, high=num_classes - 1, size=(N, 1, H, W)).astype(np.uint8)
    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
    } for _ in range(N)]
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas,
        'gt_semantic_seg': torch.LongTensor(segs)
    }
    return mm_inputs


def _prepare_input_img(img_path,
                       test_pipeline,
                       shape=None,
                       rescale_shape=None):
    # build the data pipeline
    if shape is not None:
        test_pipeline[1]['img_scale'] = (shape[1], shape[0])
    test_pipeline[1]['transforms'][0]['keep_ratio'] = False
    test_pipeline = [LoadImage()] + test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img_path)
    data = test_pipeline(data)
    imgs = data['img']
    img_metas = [i.data for i in data['img_metas']]

    if rescale_shape is not None:
        for img_meta in img_metas:
            img_meta['ori_shape'] = tuple(rescale_shape) + (3, )

    mm_inputs = {'imgs': imgs, 'img_metas': img_metas}

    return mm_inputs


def _update_input_img(img_list, img_meta_list, update_ori_shape=False):
    # update img and its meta list
    N, C, H, W = img_list[0].shape
    img_meta = img_meta_list[0][0]
    img_shape = (H, W, C)
    if update_ori_shape:
        ori_shape = img_shape
    else:
        ori_shape = img_meta['ori_shape']
    pad_shape = img_shape
    new_img_meta_list = [[{
        'img_shape':
        img_shape,
        'ori_shape':
        ori_shape,
        'pad_shape':
        pad_shape,
        'filename':
        img_meta['filename'],
        'scale_factor':
        (img_shape[1] / ori_shape[1], img_shape[0] / ori_shape[0]) * 2,
        'flip':
        False,
    } for _ in range(N)]]

    return img_list, new_img_meta_list


def pytorch2onnx(model,
                 mm_inputs,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 dynamic_export=False):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        mm_inputs (dict): Contain the input tensors and img_metas information.
        opset_version (int): The onnx op version. Default: 11.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
            Default: False.
        dynamic_export (bool): Whether to export ONNX with dynamic axis.
            Default: False.
    """
    model.cpu().eval()

    num_classes = model.num_classes

    dynamic_axes = None
    if dynamic_export:
        dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}

    # register_extra_symbolics(opset_version)
    with torch.no_grad():
        torch.onnx.export(
            model,
            mm_inputs,
            output_file,
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=show,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes)
        print(f'Successfully exported ONNX model: {output_file}')

    if verify:
        # check by onnx
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        with torch.no_grad():
            pytorch_result = model(mm_inputs)
            #pytorch_result = np.stack(pytorch_result, 0)
            pytorch_result = pytorch_result.cpu().detach().numpy()
        print(pytorch_result.shape)

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(
            None, {net_feed_input[0]: mm_inputs.detach().numpy()})[0]
        print(onnx_result.shape)

        # compare results
        np.testing.assert_allclose(
            pytorch_result.astype(np.float32) / num_classes,
            onnx_result.astype(np.float32) / num_classes,
            rtol=1e-5,
            atol=1e-5,
            err_msg='The outputs are different between Pytorch and ONNX')
        print('The outputs are same between Pytorch and ONNX')
        print(pytorch_result)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert torch model to ONNX')
    parser.add_argument('--tos_bucket', type=str,
                        help='model zoo bucket name')
    parser.add_argument('--tos_access_key', type=str,
                        help='model zoo bucket access key')
    parser.add_argument('--model_name', type=str,
                        help='model name')
    parser.add_argument('--model_version', type=str,
                        help='model version')
    parser.add_argument('--model_backend', type=str, default='haggs',
                        help='model backend, currently support haggs or timm models')
    parser.add_argument('--model_path', type=str,
                        help='model path on hdfs, http or local')
    parser.add_argument('--img_size', type=int, default=224,
                        help='model input image size, default is 224')
    parser.add_argument('--pretrained', action='store_true',
                        help='whether to use pretrained weights, default is False.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show onnx graph and segmentation results')
    parser.add_argument(
        '--verify', action='store_true', help='verify the onnx model')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether to export onnx with dynamic axis.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    tos_helper = None
    if args.tos_bucket and args.tos_access_key:
        # init model zoo and tos helper
        tos_helper = TOSHelper(
            args.tos_bucket,
            args.tos_access_key)

    model = create_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        pretrained_version=args.model_version,
        pretrained_uri=args.model_path,
        backend=args.model_backend,
        tos_helper=tos_helper,
    ).cuda()

    # prepare dimmy inputs
    dummy_inputs = torch.rand(2, 3, args.img_size, args.img_size)

    # convert model to onnx file
    pytorch2onnx(
        model,
        dummy_inputs,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        dynamic_export=args.dynamic_export)
