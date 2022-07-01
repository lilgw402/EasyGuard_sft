
"""
https://github.com/youweiliang/evit/blob/master/vision_transformer.py#L300
"""

import torch
from torch import nn

try:
    from kmeans_pytorch import kmeans
except:
    print('no kmeans_pytorch')


def not_all_patch(visual_backbone, *args, **kwargs):
    return NotAllPatch(visual_backbone, *args, **kwargs)


class NotAllPatch(nn.Module):
    def __init__(self, visual_backbone, keep_token, mode='cls', dim=768, *args, **kwargs):
        super().__init__()
        self.backbone = visual_backbone
        self.mode = mode
        self.keep_token = keep_token
        self.autoencoder = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, dim)
        )
        if self.mode == 'attn':
            self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=12, batch_first=True)
            self.query = nn.Embedding(keep_token - 1, dim)

    def forward(self, *args, **kwargs):
        if kwargs.get('return_dict'):
            kwargs.pop('return_dict')
        out = self.backbone(return_dict=True, *args, **kwargs)
        # {'feature_map': outputs, 'pooled_out': outputs[:, 0, :]} b, s, d, b, d
        if self.mode == 'cls':
            return self.cls_mode_encode(out)
        elif self.mode == 'attn':
            return self.attn_mode_encode(out)
        elif self.mode == 'cluster':
            return self.cluster_mode_encode(out)

    def cls_mode_encode(self, out):
        cls_emb = out['pooled_out']
        fm = out['feature_map']
        cls_attn = cls_emb.unsqueeze(-2) @ fm.transpose(-1, -2)  # b, s - 1
        cls_attn = cls_attn.squeeze(1)
        _, idx = torch.topk(cls_attn, self.keep_token - 1, dim=1, largest=True, sorted=True)  # [B, left_tokens-1] -1是留一个位置给fuse
        # 取 top k
        batch_size, seq_len, dim = fm.shape
        position_shift = (seq_len * torch.arange(batch_size, device=fm.device)).unsqueeze(-1)
        flat_positions = torch.reshape(idx + position_shift, [-1]).long()
        flat_sequence = torch.reshape(fm, [batch_size * seq_len, dim])
        gathered = flat_sequence.index_select(0, flat_positions)
        top_token = torch.reshape(gathered, [batch_size, -1, dim])

        # 剩下的按权重求和
        one_hot_idx = torch.nn.functional.one_hot(idx, num_classes=seq_len).sum(1)  # [bsz, token_num, seq_len]
        # comp_one_hot_idx = 1 - one_hot_idx
        weight = (one_hot_idx * (-10000) + cls_attn).unsqueeze(-1)
        weight = torch.softmax(weight, dim=1)
        extra_token = torch.sum(weight * fm, dim=1, keepdim=True)  # [bsz, 1, dim]

        visual_tokens = torch.cat([top_token, extra_token], dim=1)
        visual_tokens = self.autoencoder(visual_tokens)
        cls_emb = self.autoencoder(cls_emb)
        return {'feature_map': visual_tokens, 'pooled_out': cls_emb}

    def attn_mode_encode(self, out):
        cls_emb = out['pooled_out']
        fm = out['feature_map']
        bsz, _, dim = fm.shape
        query = self.query(torch.arange(self.keep_token - 1, device=fm.device))
        query = query.repeat(bsz, 1, 1)
        query = torch.cat([cls_emb.unsqueeze(1), query], dim=1)
        x, weight = self.attn(query=query, key=fm, value=fm)

        x = self.autoencoder(x)
        # cls_emb = self.autoencoder(cls_emb)
        return {'feature_map': x,
                'pooled_out': cls_emb,
                'attention_weight': weight}

    def cluster_mode_encode(self, out):
        cls_emb = out['pooled_out']
        fm = out['feature_map']
        bsz, _, dim = fm.shape

        cluster_tokens = []
        for i in fm:  # [seq_len, dim]
            cluster_ids, cluster_centers = kmeans(
                X=i, num_clusters=self.keep_token, distance='cosine', device=fm.device,
                tqdm_flag=False, iter_limit=5,
            )  # [seq_len]
            for c in range(self.keep_token):
                cur_cluster_index = (cluster_ids == c).unsqueeze(-1)
                cur_cluster_index = cur_cluster_index.to(i.device)
                # print(cur_cluster_index.shape, i.shape, cur_cluster_index.device, i.device)
                token_c = (i * cur_cluster_index).sum(dim=0) / cur_cluster_index.sum()  # [dim]
                cluster_tokens.append(token_c)

        cluster_tokens = torch.stack(cluster_tokens, dim=0)
        cluster_tokens = cluster_tokens.reshape([bsz, self.keep_token, dim])  # [bsz, cluster, dim]
        return {'feature_map': cluster_tokens,
                'pooled_out': cls_emb}


def complement_idx(idx, dim):
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor, find the complement for its trailing dimension,
    all other dimension is considered batched.
    Args:
        idx: input index, shape: [N, *, K]
        dim: the max index for complement
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl
