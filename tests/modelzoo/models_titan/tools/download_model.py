import argparse
from titan import TOSHelper, download_model_weights_from_tos


def parse_args():
    # Parse args with argparse tool
    parser = argparse.ArgumentParser(description='Load Model from Model Zoo')
    parser.add_argument('--tos_bucket', type=str,
                        help='model zoo bucket name')
    parser.add_argument('--tos_access_key', type=str,
                        help='model zoo bucket access key')
    parser.add_argument('--tos_model_dir', type=str,
                        help='model directory path on tos')
    parser.add_argument('--dst_model_dir', type=str,
                        help='local model directory where downloaded model will be put')
    return parser.parse_args()


def main():
    args = parse_args()

    tos_helper = TOSHelper(
        args.tos_bucket,
        args.tos_access_key)

    model_path = download_model_weights_from_tos(
        tos_helper,
        args.tos_model_dir,
        dst_dir=args.dst_model_dir)
    print(model_path)


if __name__ == '__main__':
    main()
