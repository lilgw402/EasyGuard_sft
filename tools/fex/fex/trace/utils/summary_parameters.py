# -*- coding: utf-8 -*-
'''
Created on Jan-13-21 15:28
summary_parameters.py
@author: liuzhen.nlp
Description: 
'''

def summary_parameters(model, logger=None):
    """
    Summary Parameters of Model
    :param model: torch.nn.module_name
    :param logger: logger
    :return: None
    """

    print('>> Trainable Parameters:')
    trainable_paramters = [(str(n), str(v.dtype), str(tuple(v.shape)), str(v.numel()))
                           for n, v in model.named_parameters() if v.requires_grad]
    max_lens = [max([len(item) + 4 for item in col]) for col in zip(*trainable_paramters)]
    raw_format = '|' + '|'.join(['{{:{}s}}'.format(max_len) for max_len in max_lens]) + '|'
    raw_split = '-' * (sum(max_lens) + len(max_lens) + 1)
    print(raw_split)
    print(raw_format.format('Name', 'Dtype', 'Shape', '#Params'))
    print(raw_split)

    for name, dtype, shape, number in trainable_paramters:
        print(raw_format.format(name, dtype, shape, number))
        print(raw_split)

    num_trainable_params = sum([v.numel() for v in model.parameters() if v.requires_grad])
    total_params = sum([v.numel() for v in model.parameters()])
    non_trainable_params = total_params - num_trainable_params
    print('>> {:25s}\t{:.2f}\tM'.format('# TrainableParams:', num_trainable_params / (1.0 * 10 ** 6)))
    print('>> {:25s}\t{:.2f}\tM'.format('# NonTrainableParams:', non_trainable_params / (1.0 * 10 ** 6)))
    print('>> {:25s}\t{:.2f}\tM'.format('# TotalParams:', total_params / (1.0 * 10 ** 6)))
