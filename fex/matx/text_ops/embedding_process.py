# -*- coding: utf-8 -*-

from typing import List, Tuple
from matx.runtime.ndarray import NDArray
import torch
from torch._C import dtype
from fex.matx.text_ops.pad_sequence import pad_sequence
import matx
from torch.nn import functional as F


class EmbedProcess(torch.nn.Module):

    def __init__(self, image_token_num: int = 8, image_feature_dim: int = 128):
        super().__init__()
        self.image_token_num: int = image_token_num
        self.image_feature_dim: int = image_feature_dim

    def forward(self, image_embs: List[torch.Tensor], with_query: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 将image_embs根据image_token_num和image_feature_dim做padding操作。同时根据with_query参数，决定是否在第一个位置加上query对应的mock embed。
        这个逻辑主要是因为train和serving的过程不一致导致的。train是qt和qtv分别过bert，serving将这个过程合起来一次过bert，减少计算量。

        Args:
            image_embs (List[torch.Tensor]): 图片编码embeds.
            with_query (bool, optional): 是否在第一个位置加上query对应的mock embed. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: pad_image_embs 和 image_masks.
        """
        image_emb_type = image_embs[0].dtype
        image_emb_device = image_embs[0].device

        batch_vis_embs = []
        # batch_vis_masks = []

        if with_query:
            batch_vis_embs.append(torch.zeros(size=(self.image_token_num, self.image_feature_dim), dtype=image_emb_type, device=image_emb_device).float())

        for image_emb in image_embs:
            if len(image_emb.shape) != 2 or image_emb.shape[1] != self.image_feature_dim:    # 为了触发stack操作
                image_emb = torch.zeros(size=(self.image_token_num, self.image_feature_dim), dtype=image_emb_type, device=image_emb_device, pin_memory=True)
            elif image_emb.shape[0] < self.image_token_num:  # 不足8帧时pad
                pad_length = self.image_token_num - image_emb.shape[0]
                image_emb = F.pad(image_emb, (0, 0, 0, pad_length), "constant", 0.0)

            batch_vis_embs.append(image_emb.float())
        batch_vis_embs_tensor = pad_sequence(batch_vis_embs, padding_value=0.0, max_len=self.image_token_num)
        batch_vis_masks_tensor = batch_vis_embs_tensor.mean(dim=-1).ne(0).to(torch.int64, non_blocking=True)
        # batch_vis_masks_tensor = torch.ones(batch_vis_embs_tensor.shape[:-1],
        #                                     dtype=torch.long,
        #                                     device='cuda')
        return batch_vis_embs_tensor, batch_vis_masks_tensor


class EmbedProcessMatx:
    '''
    matx版本，尽量避免变量复制，如果shape不对，会自动全部mask掉
    '''

    def __init__(self, image_token_num: int = 8, image_features: int = 128) -> None:
        self.image_token_num: int = image_token_num
        self.image_features: int = image_features

        empty_masks: List = []
        for i in range(image_token_num):
            empty_masks.append(0)
        self.empty_masks: matx.NDArray = matx.NDArray(empty_masks, [image_token_num], 'int64')

        full_masks: List = []
        for i in range(image_token_num):
            full_masks.append(1)
        self.full_masks: matx.NDArray = matx.NDArray(full_masks, [image_token_num], 'int64')

        empty_emb: List = []
        for i in range(image_token_num):
            tmp: List = []
            for j in range(image_features):
                tmp.append(0.0)
            empty_emb.append(tmp)
        # self.empty_emb: matx.NDArray = matx.NDArray([], [self.image_token_num, self.image_features], 'float32')
        self.empty_emb: matx.NDArray = matx.NDArray(empty_emb, [], 'float32')

    def __call__(self, image_embs: List[matx.NDArray], with_query: int = 0) -> Tuple[matx.NDArray, matx.NDArray]:
        masks: matx.List = matx.List()
        if with_query:
            masks.reserve(len(image_embs) + 1)
            image_embs = [matx.nd_rand([self.image_token_num, 10])] + image_embs
        else:
            masks.reserve(len(image_embs))

        for i in range(len(image_embs)):
            emb = image_embs[i]
            # 扔掉不足8帧的img emb
            # if len(emb.shape()) != 2 or emb.shape()[0] != self.image_token_num or emb.shape()[1] != self.image_features:
            #     image_embs[i] = self.empty_emb
            #     masks.append(self.empty_masks)
            # else:
            #     masks.append(self.full_masks)

            # padding不足8帧的img emb
            shape = emb.shape()
            padding_len = 0
            if len(shape) != 2 or shape[1] != self.image_features:
                emb = self.empty_emb
                padding_len = self.image_token_num
            elif shape[0] != self.image_token_num:
                padding_len = self.image_token_num - shape[0]
                # default value will create nan in tensor
                # padding_array = matx.NDArray([], [padding_len, self.image_features], 'float32')
                padding_array = matx.nd_rand([padding_len, self.image_features])  # 默认为float32
                padding_array = matx.nd_sub(padding_array, padding_array)
                emb = matx.nd_concatenate([emb, padding_array], 0)
            else:
                padding_len = 0

            tmp_mask = []
            for index in range(self.image_token_num - padding_len):
                tmp_mask.append(1)
            for index in range(padding_len):
                tmp_mask.append(0)

            masks.append(matx.NDArray(tmp_mask, [self.image_token_num], 'int64'))
            image_embs[i] = emb

        batch_image_embs = matx.nd_stack(image_embs)
        batch_masks = matx.nd_stack(masks)
        return batch_image_embs, batch_masks


if __name__ == '__main__':
    import time
    import numpy as np
    from tqdm import tqdm
    import json
    from aweme_cm.utils.utils import string2array
    # model = EmbedProcess()
    # mock_data = [torch.rand(8, 128)] * 512

    # for _ in tqdm(range(100)):
    #     rs = model(mock_data)

    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
    #                                      profile_memory=False) as prof:
    #     with torch.autograd.profiler.record_function("model_inference"):
    #         t0 = time.time()
    #         outputs = model(mock_data, with_query=0)
    #         torch.cuda.synchronize()
    #         print('=' * 30, time.time() - t0)

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    # prof.export_chrome_trace('/data00/yejiandong/tmp/embed_process_profile.json')

    data = []
    with open("/data00/yejiandong/tmp/test_rawrank_data.json") as f:
        for line in f:
            query_docs = json.loads(line)
            docs = query_docs['docs']
            for d in docs:
                data.append(string2array(d['img_emb']))

    bsz = 128
    mock_data = []
    for d in data:
        arr = matx.NDArray([], d.shape, 'float32')
        arr.from_numpy(d)
        mock_data.append(arr)

    print(len(data))

    # img_emb = np.random.rand(8, 128)
    # arr1 = matx.NDArray([], img_emb.shape, str(img_emb.dtype))
    # arr1.from_numpy(img_emb)

    # img_emb2 = np.random.rand(4, 128)
    # arr2 = matx.NDArray([], img_emb2.shape, str(img_emb2.dtype))
    # arr2.from_numpy(img_emb2)

    emb, masks = EmbedProcessMatx()(mock_data)
    emb = torch.from_numpy(emb.asnumpy())
    masks = torch.from_numpy(masks.asnumpy())

    emb2, masks2 = EmbedProcess()([torch.tensor(x) for x in data])

    from rich.console import Console
    console = Console()
    console.print(emb == emb2)
    console.print(masks == masks2)
