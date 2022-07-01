#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Huang Wenguan (huangwenguan@bytedance.com)
@Date: 2020-03-02 19:51:00
@LastEditTime: 2020-05-19 22:36:24
@LastEditors: Huang Wenguan
@Description: 
'''

import sys
import numpy as np
import json
import random
import os

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(
            np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg


def calc_ndcg(labels, scores, k=None):
    new_rank = sorted(zip(scores, range(len(scores))), key=lambda x: x[0],
                      reverse=True)
    new_labels = []
    for score, idx in new_rank:
        new_labels.append(labels[idx])
    if isinstance(k, int):
        return ndcg_at_k(new_labels, k)
    else:
        ndcg1 = ndcg_at_k(new_labels, 1)
        ndcg3 = ndcg_at_k(new_labels, 3)
        ndcg4 = ndcg_at_k(new_labels, 4)
        ndcg5 = ndcg_at_k(new_labels, 5)
        return ndcg1, ndcg3, ndcg4, ndcg5

def cal_random_ndcg(fn):
    score_tag = 'trans_match_0827'
    #score_tag = 'douyin_knrm_0809'
    ndcg1_all, ndcg3_all, ndcg4_all, ndcg5_all = [], [], [], []
    if os.path.isdir(fn):
        for fi in os.listdir(fn):
            with open(os.path.join(fn, fi)) as f:
               for l in f:
                jl = json.loads(l)
                labels = [int(d['label']) for d in jl['rs']]
                #scores = [random.random() for _ in labels]
                #scores = [1/(i+1) for i in range(len(labels))]
                scores = [float(d['score']) for d in jl['rs']]
                print(scores)
                n1, n3, n4, n5 = calc_ndcg(labels, scores)
                if len(labels) > 1:
                    ndcg1_all.append(n1)
                if len(labels) > 3:
                    ndcg3_all.append(n3)
                if len(labels) > 4:
                    ndcg4_all.append(n4)
                if len(labels) > 5:
                    ndcg5_all.append(n5)
    else:
        with open(fn) as f:
            setname = 'rs'
            for l in f:
                jl = json.loads(l)
                labels = [int(d['text_label']) if d['text_label']!='x' else int(d['label']) for d in jl[setname]]
                labels = [int(d['label']) for d in jl[setname]]
                #labels = [int(d['visual_label']) if d['visual_label']!='x' else int(d['video_label']) for d in jl[setname]]
                #scores = [random.random() for _ in labels]
                scores = [float(d[score_tag]) for d in jl[setname]]
                #print(scores)
                n1, n3, n4, n5 = calc_ndcg(labels, scores)
                if len(labels) > 0:
                    ndcg1_all.append(n1)
                if len(labels) > 0:
                    ndcg3_all.append(n3)
                if len(labels) > 0:
                    ndcg4_all.append(n4)
                if len(labels) > 0:
                    ndcg5_all.append(n5)
    print('ndcg@1 ', round(sum(ndcg1_all)/len(ndcg1_all), 4), len(ndcg1_all))
    print('ndcg@3 ', round(sum(ndcg3_all)/len(ndcg3_all), 4), len(ndcg3_all))
    print('ndcg@4 ', round(sum(ndcg4_all)/len(ndcg4_all), 4), len(ndcg4_all))
    print('ndcg@5 ', round(sum(ndcg5_all)/len(ndcg5_all), 4), len(ndcg5_all))

if __name__ == "__main__":
    fn = sys.argv[1] # '/data00/huangwenguan/score_0'
    cal_random_ndcg(fn)
