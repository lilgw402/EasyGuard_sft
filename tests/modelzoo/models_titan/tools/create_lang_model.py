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
    parser.add_argument('--input_length', type=int, default=32,
                        help='input lenght')
    parser.add_argument('--pretrained', action='store_true',
                        help='whether to use pretrained weights, default is False.')
    parser.add_argument('--embedder_only', action='store_true',
                        help='if True, only use the embedding step.')
    parser.add_argument('--mlm_enable', action='store_true',
                        help='if True, use Bert MLM for training.')
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
        tos_helper=tos_helper,
        embedder_only=args.embedder_only,
        mlm_enable=args.mlm_enable,
    ).cuda()

    # prepare data
    input_ids = torch.randint(10000, (2, args.input_length)).cuda()
    attn_masks = torch.ones((2, args.input_length)).cuda()

    # model forward to get results
    if args.embedder_only:
        feats, attn_out_masks = model(input_ids, attn_masks)
        print(feats.shape)
        print(attn_out_masks.shape)
    else:
        feats = model(input_ids, attn_masks)
        print(feats.shape)

# Outputs with embedder_only (features and attention_masks):
#   torch.Size([2, 32, 768])
#   torch.Size([2, 32])
# Outputs only features:
#   torch.Size([2, 768])


if __name__ == '__main__':
    main()
