import argparse
import torch
from titan import TOSHelper, create_model


def parse_args():
    # Parse args with argparse tool
    parser = argparse.ArgumentParser(description='Load Model from Model Zoo')
    parser.add_argument('--model_name', type=str, required=True, help='model name')
    parser.add_argument('--tos_bucket', type=str,
                        help='model zoo bucket name')
    parser.add_argument('--tos_access_key', type=str,
                        help='model zoo bucket access key')
    parser.add_argument('--tos_idc', type=str, default='',
                        help='model zoo bucket idc, for sg bucket, please specify sgcomm1')
    parser.add_argument('--model_version', type=str,
                        help='model version')
    parser.add_argument('--model_backend', type=str, default='titan',
                        help='model backend, currently support titan or timm models')
    parser.add_argument('--model_path', type=str,
                        help='model path on hdfs, http or local')
    parser.add_argument('--input_shape', type=str, default='3,224,224',
                        help='model input image shape, string split by comma. Default is 3,224,224')
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='Number of classes, default is 1000.')
    parser.add_argument('--pretrained', action='store_true',
                        help='whether to use pretrained weights, default is False.')
    parser.add_argument('--features_only', action='store_true',
                        help='whether to only output features from the model, default is False.')
    return parser.parse_args()


def main():
    args = parse_args()

    tos_helper = None
    if args.tos_bucket and args.tos_access_key:
        # init tos helper
        tos_helper = TOSHelper(
            args.tos_bucket,
            args.tos_access_key,
            idc=args.tos_idc)

    # create model
    model = create_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        pretrained_version=args.model_version,
        pretrained_uri=args.model_path,
        backend=args.model_backend,
        features_only=args.features_only,
        num_classes=args.num_classes,
        tos_helper=tos_helper,
    ).cuda()

    # prepare data
    input_shape = list(map(int, args.input_shape.split(',')))
    # extend input_shape with batch_size=2
    input_shape.insert(0, 2)
    torch.manual_seed(2022)
    data_batch = torch.rand(input_shape).cuda()

    # model forward to get results
    out = model(data_batch)
    # for transformer models with use_attn_map=True
    if isinstance(out, tuple):
        print('attn map:', out[1].shape)
        out = out[0]

    if isinstance(out, list):
        for o in out:
            print(o.shape)
    else:
        print(out.shape)


# Outputs with fc:
#   torch.Size([2, 1000])
# Features_only outputs:
#   torch.Size([2, 256, 56, 56])
#   torch.Size([2, 512, 28, 28])
#   torch.Size([2, 1024, 14, 14])
#   torch.Size([2, 2048, 7, 7])


if __name__ == '__main__':
    main()
