import argparse
import torch
from ptflops import get_model_complexity_info
from titan import TOSHelper, create_model


def parse_args():
    # Parse args with argparse tool
    parser = argparse.ArgumentParser(description='Load Model from Model Zoo')
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
    parser.add_argument('--input_shape', type=str, default='3,224,224',
                        help='model input image shape, string split by comma. Default is 3,224,224')
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
            args.tos_access_key)

    # create model
    model = create_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        pretrained_version=args.model_version,
        pretrained_uri=args.model_path,
        backend=args.model_backend,
        features_only=args.features_only,
        tos_helper=tos_helper,
    )

    input_shape = tuple(map(int, args.input_shape.split(',')))
    macs, params = get_model_complexity_info(model, input_shape, as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == '__main__':
    main()
