#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import torch

from fex.config import cfg
from fex.utils.load import load_from_pretrain
from fex.nn.backbone.conv_next import convnext
from example.image_classify.model import ResNetClassifier


from fex.data.benchmark.imagenet import get_imagenet_dataloader


def run(model_type="resnet50"):
    """ run evaluation of imagenet """
    model = get_model(model_type)
    run_benchmark(model)


def get_model(model_type):
    config_path = 'example/image_classify/resnet_json.yaml'
    cfg.update_cfg(config_path)
    if model_type == "resnet50":
        model = ResNetClassifier(cfg)
        ckpt = 'hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/vl/model/' \
               'pretrained_model/resnet50-19c8e357-t3.pth'
        #ckpt = 'hdfs://haruna/byte_search/user/shidai.0919/video_label_date/fex_image_classify/model_state_epoch_300240.th'
    elif model_type == "convnext_tiny":
        model, ckpt = convnext("tiny")
        model = ResNetClassifier(cfg, model=model)
    elif model_type == "convnext_base":
        model, ckpt = convnext("base", in_22k=True)
        model = ResNetClassifier(cfg, model=model)
    else:
        raise NotImplementedError()

    #load_from_pretrain(model, ckpt, [])
    load_from_pretrain(model, ckpt, ['->resnet.'])
    model.eval()

    if torch.cuda.is_available():
        model.to('cuda')
    return model


def run_benchmark(model):
    model.eval()
    # 2. data
    # folder 版本
    #source, preprocess_type, data_path = 'folder', 'torchvision', '/mnt/bd/search-mm-space/data/research/imagenet/val'
    # hdfs torchvision 版
    source, preprocess_type, data_path = 'hdfs', 'torchvision', 'hdfs://haruna/home/byte_search_nlp_lq/multimodal/data/academic/imagenet/val_json'
    # hdfs dali 版
    # source, preprocess_type, data_path = 'hdfs', 'dali', 'hdfs://haruna/home/byte_search_nlp_lq/multimodal/data/academic/imagenet/val_json'
    # hdfs bytedvision 版
    #source, preprocess_type, data_path = 'hdfs', 'bytedvision', 'hdfs://haruna/home/byte_search_nlp_lq/multimodal/data/academic/imagenet/val_json'

    val_loader = get_imagenet_dataloader(
        data_path=data_path,
        batch_size=512,
        source=source,
        preprocess_type=preprocess_type,
        mode='val'
    )

    # 3. evaluate
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    total = 0
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            if isinstance(data, dict):
                images = data['image']
                target = data['label']
            elif isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
                data = data[0]
                images = data['image']
                target = data['label']
            else:
                images, target = data
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

            if i % 100 == 0:
                print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}. Total {total}. Time {tm.sum:.3f}'
                      .format(top1=top1, top5=top5, total=total, tm=batch_time))

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}. Total {total}. Time {tm.sum:.3f}'
              .format(top1=top1, top5=top5, total=total, tm=batch_time))


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
    st = time.time()
    run()
    print('use time: ', time.time() - st)
