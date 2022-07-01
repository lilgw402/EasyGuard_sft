#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
这个脚本是用来评估imagenet的分类指标：
1. 一个是在fc层出来的指标
2. 一个是将label当做文本来算embedding，看cosine的指标
'''
from absl import flags
from absl import app

import os
import time
import json
import random

# import tensorflow as tf
# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

import torch
from torchvision import transforms

from fex.data import KVSampler, worker_init_fn, BertTokenizer
from fex.utils.load import load_from_pretrain
from fex.config import cfg

from example.clip.model import CLIPNet
from example.clip.dataset_imagenet import ImageNetKVDataset, load_labelmapping

FLAGS = flags.FLAGS


def def_flags():
    flags.DEFINE_string("config_path", "hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/clue/abvt_b32_tnb_fpt_018b_lp_04/abvt_b32tnb_fpt_018b_lp.yaml", "config path")
    flags.DEFINE_string("ckpt", "hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/clue/abvt_b32_tnb_fpt_018b_lp_04/model_state_epoch_226687.th", "checkpoint")
    flags.DEFINE_string("output_path", "imagenet_output", "output")


def gen_loader(cfg, data_path):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    print(local_rank, 'local_rank')
    torch.cuda.set_device(local_rank)
    num_readers = 8
    batch_size = 128
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset = ImageNetKVDataset(cfg, data_path, num_readers, transform)
    val_sampler = KVSampler(val_dataset,
                            batch_size=batch_size,
                            num_replicas=1,
                            rank=0,
                            shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=None,  # 这里要 None, 因为batching是在Dataset里做
                                             sampler=val_sampler,
                                             num_workers=4,
                                             worker_init_fn=worker_init_fn,
                                             drop_last=False)
    return val_loader


def gen_text_embedding(cfg, model, do_aug=True, use_english=False):
    """
    do_aug: 如果 set 为 true，则会增加一些增强，变成："xxx 的 图片"
    """
    print('calculating label embedding')
    label_embs = []
    label_tags = []
    tag2label = load_labelmapping()
    sortedlabelinfo = sorted(tag2label.values(), key=lambda x: x['label_idx'])
    print(sortedlabelinfo[:10])

    tokenizer = BertTokenizer(cfg.network.vocab_file, do_lower_case=True)

    for info in sortedlabelinfo:
        if use_english:
            name = info['english_name'].replace(',', '')
        else:
            name = info['chinese_name']
        label_tags.append(name)
        if do_aug:
            name = '%s 的 图片' % name
        tokens = tokenizer.tokenize(name)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids).cuda().unsqueeze(0)
        input_segment_ids = torch.zeros_like(input_ids)
        input_mask = torch.ones_like(input_ids)
        t_emb = model.encode_text(input_ids=input_ids, input_segment_ids=input_segment_ids, input_mask=input_mask)[0]
        label_embs.append(t_emb)

    label_embs = torch.stack(label_embs, dim=0)
    print('loaded label embedding, %s, %s, %s' % (label_embs.shape, len(label_tags), label_tags[:10]))
    return label_tags, label_embs


@torch.no_grad()
def run_validation_benchmark(config_path, ckpt, output_path):
    """ run evaluation of imagenet """
    # 1. model
    cfg.update_cfg(config_path)
    max_len = 16
    model = CLIPNet(config=cfg)
    load_from_pretrain(model, ckpt, [])
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    # 2. data
    data_path = "hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/imagenet/val"
    val_loader = gen_loader(cfg, data_path)

    # 3. label embedding
    label_tags, label_embedding = gen_text_embedding(cfg, model)  # [1000, dim]

    # evaluate
    batch_time = AverageMeter('Time', ':6.3f')
    emb_top1 = AverageMeter('EmbAcc@1', ':6.2f')
    emb_top5 = AverageMeter('EmbAcc@5', ':6.2f')

    total = 0
    with hdfs_open(os.path.join(output_path, 'imagenet_val.emb'), 'w') as fo, \
            hdfs_open(os.path.join(output_path, 'result.txt'), 'a') as f, \
            hdfs_open(os.path.join(output_path, 'imagenet_val_sample.txt'), 'w') as fsp:

        end = time.time()
        for i, data in enumerate(val_loader):
            target = data['label']
            names = data.pop('names')
            index = data.pop('index')
            label = data.pop('label')
            for k, v in data.items():
                data[k] = v.cuda(non_blocking=True)
            total += len(target)

            # compute output
            output = model.encode_image(data['image'])

            acc1_cnt = 0.
            acc5_cnt = 0.
            v_emb = output  # ['pooled_out']
            for j, (name, idx) in enumerate(zip(names, index)):
                cur_emb = v_emb[j]
                top5idx, top5dist = closest_with_emb(label_embedding, cur_emb)

                if label_tags.index(name) == top5idx[0]:
                    acc1_cnt += 1
                if label_tags.index(name) in top5idx:
                    acc5_cnt += 1
                to_write_str = json.dumps({'cur_emb': ' '.join([str(x) for x in cur_emb.tolist()]),
                                           'name': name,
                                           'index': idx}, ensure_ascii=False) + '\n'
                if output_path.startswith('hdfs'):
                    to_write_str = to_write_str.encode()
                fo.write(to_write_str)

                # 随机写一些top cosine，算是visualize
                if random.random() < 0.002:
                    to_write_str = 'image: %s, gt: %s, top5: %s\n' % (idx, name, [label_tags[ti] for ti in top5idx])
                    if output_path.startswith('hdfs'):
                        to_write_str = to_write_str.encode()
                    fsp.write(to_write_str)

            emb_top1.update(acc1_cnt / len(target) * 100, len(target))
            emb_top5.update(acc5_cnt / len(target) * 100, len(target))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        emb_top1_avg = torch.tensor(emb_top1.avg)
        emb_top5_avg = torch.tensor(emb_top5.avg)
        total = torch.tensor(total)
        # emb_top1_avg = sync_tensor(emb_top1_avg, reduce_op='avg')
        # emb_top5_avg = sync_tensor(emb_top5_avg, reduce_op='avg')
        # total = sync_tensor(total).tolist()
        emb_top1_avg = round(emb_top1_avg.tolist(), 4)
        emb_top5_avg = round(emb_top5_avg.tolist(), 4)

        print(' * Acc@1 %s Acc@5 %s. Total %s' % (emb_top1_avg, emb_top5_avg, total))

        s = '[IMAGENET] total: %s\n' % total
        s += 'Emb\nAcc@1 %s Acc@5 %s\n' % (emb_top1_avg, emb_top5_avg)
        if output_path.startswith('hdfs'):
            s = s.encode()
        f.write(s)

    return emb_top1, emb_top5


def closest_with_emb(e, cur):
    dist = torch.cosine_similarity(cur.unsqueeze(0), e)
    index_sorted = torch.argsort(dist)
    top5 = index_sorted[-5:].tolist()
    top5 = top5[::-1]
    return top5, dist[top5]


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


from fex.utils.hdfs_io import hopen


def hdfs_open(f, mode='r'):
    if f.startswith('hdfs'):
        return hopen(f, mode)
    else:
        return open(f, mode)


def main(_):
    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)
    run_validation_benchmark(FLAGS.config_path, FLAGS.ckpt, FLAGS.output_path)


if __name__ == "__main__":
    def_flags()
    app.run(main)
