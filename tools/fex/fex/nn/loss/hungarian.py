""" hungarian """

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F


class HungarianMatcherCE(nn.Module):
    """
    see https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/models/matcher.py#L85
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets):
        """匹配分类结果

        Args:
            outputs ([torch.Tensor]): [batch_size, num_queries, num_classes]
            targets ([torch.Tensor]): [batch_size, num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)


        """
        bs, num_queries = outputs.shape[:2]

        out_prob = outputs.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        tgt_ids = torch.cat(targets)  # sum(batch_size, num_target_boxes)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        C = -out_prob[:, tgt_ids]
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v) for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class HungarianMatcherCOS(nn.Module):
    """
    see https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/models/matcher.py#L85
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets):
        """匹配，按cosine距离作为度量

        Args:
            outputs ([torch.Tensor]): [batch_size, num_queries, dim]
            targets ([torch.Tensor]): [batch_size, torch.Tensor(num_target, dim)] (where num_target is the number of ground-truth
                           objects in the target) containing the class labels
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)


        """
        bs, num_queries = outputs.shape[:2]

        out_vec = outputs.flatten(0, 1)  # [batch_size * num_queries, dim]
        out_vec = F.normalize(out_vec, dim=-1)

        tgt_vec = torch.cat(targets)  # [sum(batch_size, num_target_boxes), dim]
        tgt_vec = F.normalize(tgt_vec, dim=-1)

        # Compute the similarity loss, we use - dot product
        C = - torch.matmul(out_vec, tgt_vec.t())
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v) for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class HungarianMatcherCOSBatch(nn.Module):
    """
    see https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/models/matcher.py#L85
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets):
        """匹配，按cosine距离作为度量
        和上面的区别是，targets 这里是定长的，不再是list

        Args:
            outputs ([torch.Tensor]): [batch_size, num_queries, dim]
            targets ([torch.Tensor]): [batch_size, num_target, dim] (where num_target is the number of ground-truth
                           objects in the target) containing the class labels
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)


        """
        bs, num_queries = outputs.shape[:2]

        out_vec = outputs.flatten(0, 1)  # [batch_size * num_queries, dim]
        out_vec = F.normalize(out_vec, dim=-1)

        bs, num_target = targets.shape[:2]
        tgt_vec = targets.flatten(0, 1)  # [batch_size * num_target_boxes, dim]
        tgt_vec = F.normalize(tgt_vec, dim=-1)

        # Compute the similarity loss, we use - dot product
        C = - torch.matmul(out_vec, tgt_vec.t())
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [num_target] * bs

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
