""" 集合 loss，预测是一个集合，target也是一个集合 """

import torch
from torch import nn
import torch.nn.functional as F

from fex.nn.loss import HungarianMatcherCE, HungarianMatcherCOS, HungarianMatcherCOSBatch


class SetCriterionCE(nn.Module):
    """see https://github.com/facebookresearch/detr/blame/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/models/detr.py#L83
    先做一次hungarian matching assignment，把ground truth label 和 model prediction 做一一匹配
    然后一一算loss。
    只支持分类的情况，用交叉熵算loss
    """

    def __init__(self, num_classes, eos_coef):
        """create

        Args:
            num_classes (int): 分类数目
            eos_coef (int): relative classification weight applied to the no-object category
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcherCE()

        # 其他位置都是1的权重，最后一个no-object category的权重是eos_coef
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, targets):

        # 匹配
        indices = self.matcher(outputs, targets)

        # 算loss
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(outputs.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=outputs.device)
        target_classes[idx] = target_classes_o
        loss = F.cross_entropy(outputs.transpose(1, 2), target_classes, self.empty_weight)

        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


class SetCriterionCOS(nn.Module):
    """see https://github.com/facebookresearch/detr/blame/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/models/detr.py#L83
    先做一次hungarian matching assignment，把ground truth label 和 model prediction 做一一匹配
    然后一一算loss。
    用cosine sim作为loss
    """

    def __init__(self, dim, eos_coef, matcher='HungarianMatcherCOS'):
        """create

        Args:
            num_classes (int): 分类数目
            eos_coef (int): relative classification weight applied to the no-object category
        """
        super().__init__()
        if matcher == 'HungarianMatcherCOS':
            self.matcher = HungarianMatcherCOS()
        elif matcher == 'HungarianMatcherCOSBatch':
            self.matcher = HungarianMatcherCOSBatch()

        self.dim = dim
        # 其他位置都是1的权重，最后一个no-object category的权重是eos_coef
        self.eos_coef = eos_coef
        self.empty_vec = torch.nn.Parameter(torch.empty(1, self.dim))
        nn.init.xavier_normal_(self.empty_vec)

    def forward(self, outputs, targets):

        # 匹配
        indices = self.matcher(outputs, targets)

        # 构造target vec，index的地方是匹配上的vec，其他地方是empty vec
        idx = self._get_src_permutation_idx(indices)
        target_vec_o = torch.cat([t[J] for t, (_, J) in zip(targets, indices)])
        expaned_shape = list(outputs.shape[:2]) + [self.dim]
        target_vec = self.empty_vec.expand(expaned_shape).clone()  # TODO: 不知道会不会很慢
        # 这个index的逻辑是，先对tuple第一个元素，做batch level的取值，再对tuple第二个原始，做vec level的取值
        # 比如 idx=(tensor([0, 1, 1]), tensor([2, 1, 2]))，则一共取3个元素：batch0第2个、batch1第1个、batch1第2个
        target_vec[idx] = target_vec_o

        # 算 loss
        outputs = F.normalize(outputs, dim=-1)
        target_vec = F.normalize(target_vec, dim=-1)
        loss = 1 - torch.mul(outputs, target_vec).sum(-1)
        eos_mask = torch.full(loss.shape, self.eos_coef, device=loss.device)
        eos_mask[idx] = 1.
        loss = loss * eos_mask
        loss = loss.mean()

        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
