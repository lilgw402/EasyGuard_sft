import argparse
import torch
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
    parser.add_argument('--img_size', type=int, default=224,
                        help='model input image size, default is 224')
    parser.add_argument('--optimal_bz', type=int, default=32,
                        help='optimal batch size that can fit in gpu device, default is 32')
    parser.add_argument('--pretrained', action='store_true',
                        help='whether to use pretrained weights, default is False.')
    return parser.parse_args()


def main():
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

    # prepare data
    optimal_batch_size = args.optimal_bz
    dummy_input = torch.rand(optimal_batch_size, 3,
                             args.img_size, args.img_size).cuda()

    # measure performance
    repetitions = 100
    total_time = 0
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(
                enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * optimal_batch_size) / total_time
    print('Final Throughput:', Throughput)


if __name__ == '__main__':
    main()
