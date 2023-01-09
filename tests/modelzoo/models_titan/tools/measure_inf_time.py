import argparse
import torch
import numpy as np
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
    dummy_input = torch.rand(1, 3, args.img_size, args.img_size).cuda()

    # init loggers
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings = np.zeros((repetitions, 1))

    # gpu warm-up
    for _ in range(10):
        _ = model(dummy_input)

    # measure performance
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # wait for gpu sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn, std_syn)


if __name__ == '__main__':
    main()
