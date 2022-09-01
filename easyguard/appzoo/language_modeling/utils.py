# -*- coding: utf-8 -*-

import torch
from cruise.utilities.cloud_io import load as crs_load
from cruise.utilities.rank_zero import rank_zero_info, rank_zero_warn


def load_pretrained(load_pretrain, model):
    rank_zero_info(f"==============> Loading weight {load_pretrain} for fine-tuning......")
    checkpoint = crs_load(load_pretrain, map_location='cpu')
    state_dict = checkpoint

    # check classifier, if not match, then re-init classifier to zero
    try:
        head_bias_pretrained = state_dict['classifier.bias']
        Nc1 = head_bias_pretrained.shape[0]
        Nc2 = model.classifier.bias.shape[0]
        if (Nc1 != Nc2):
            torch.nn.init.constant_(model.classifier.bias, 0.)
            torch.nn.init.constant_(model.classifier.weight, 0.)
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            rank_zero_warn("Error in loading classifier head, re-init classifier head to 0")
    except:
        rank_zero_warn("Error in loading classifier weights...")
    msg = model.load_state_dict(state_dict, strict=False)
    rank_zero_warn(str(msg))

    rank_zero_info(f"=> loaded successfully '{load_pretrain}'")

    del checkpoint
    torch.cuda.empty_cache()
