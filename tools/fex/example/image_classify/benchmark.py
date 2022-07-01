#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import torch
from torchvision import transforms

from fex.config import cfg
from fex.utils.load import load_from_pretrain
from fex.nn.backbone.conv_next import convnext
from example.image_classify.imagenet_json_dataset import ImageNetJsonDataset, get_transform
from example.image_classify.model import ResNetClassifier


def run_benchmark(model_type="resnet50"):
    """ run evaluation of imagenet """
    # 1. model
    config_path = 'example/image_classify/resnet_json.yaml'
    cfg.update_cfg(config_path)
    if model_type == "resnet50":
        model = ResNetClassifier(cfg)
        ckpt = 'hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/vl/model/' \
               'pretrained_model/resnet50-19c8e357-t3.pth'
    elif model_type == "convnext_tiny":
        model, ckpt = convnext("tiny")
        model = ResNetClassifier(cfg, model=model)
    elif model_type == "convnext_base":
        model, ckpt = convnext("base", in_22k=True)
        model = ResNetClassifier(cfg, model=model)
    else:
        raise NotImplementedError()

    load_from_pretrain(model, ckpt, ['->resnet.'])
    model.eval()

    if torch.cuda.is_available():
        model.to('cuda')

    # 2. data
    data_path = "hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/data/tower/imagenet_val_nte_org"
    val_transform = get_transform(mode="val")

    val_dataset = ImageNetJsonDataset(data_path,
                                      transform=val_transform,
                                      rank=int(os.environ.get('RANK') or 0),
                                      world_size=int(
                                          os.environ.get('WORLD_SIZE') or 1))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=512,
                                             num_workers=8,
                                             pin_memory=True,
                                             drop_last=False,
                                             collate_fn=val_dataset.collect_fn)

    # 3. evaluate
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    total = 0
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images = data['image']
            target = data['label']
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            total += len(target)
            # compute output
            output = model(image=images, label=target)
            # measure accuracy and record loss
            if isinstance(output, dict):
                output = output['scores']
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}. Total {total}'
                      .format(top1=top1, top5=top5, total=total))

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}. Total {total}'
              .format(top1=top1, top5=top5, total=total))


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    run_benchmark()
