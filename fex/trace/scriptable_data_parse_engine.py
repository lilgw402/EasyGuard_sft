# -*- coding: utf-8 -*-
'''
Created on Jan-13-21 11:53
scriptable_preprocess_engine.py
@author: liuzhen.nlp
Description: 
'''

import torch

class ScriptableDataParseEngine(torch.nn.Module):
    """
    一个可以Trace的DataParse基类, DataParse需要继承该类, 重写forward方法来实现自定义的预处理流程.
    TODO: 将matx4中的一些预处理逻辑加入进来
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        """[summary]

        Returns:
            [type]: [description]
        """
        raise ValueError("Forward method not implement in ScriptableDataParseEngine, please check! ")

