
import torch
from fex.utils.torch_io import load


def reconstruct_pos(ckpt, target_path):

    infer_map_height = {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
        4: 2,
        5: 3,
        6: 4,
        7: 4,
        8: 5,
        9: 5,
        10: 6,
        11: 6
    }
    infer_map_width = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5
    }

    model = load(ckpt)
    pos_emb = model['resnet.positional_embedding']
    cls_emb = pos_emb[0].unsqueeze(0)
    res_emb = pos_emb[1:, :]
    target_emb = torch.zeros([72, 768], device=pos_emb.device)
    for i in range(12):
        for j in range(6):
            origin_idx_h = infer_map_height[i]
            origin_idx_w = infer_map_width[j]
            origin_idx = origin_idx_h * 7 + origin_idx_w
            target_idx = i * 6 + j
            print('target: %s = %s %s, from: %s = %s %s' % (target_idx, i, j, origin_idx, origin_idx_h, origin_idx_w))
            target_emb[target_idx] = res_emb[origin_idx]

    print(target_emb.device, cls_emb.device)
    target_emb = torch.cat([cls_emb, target_emb], dim=0)
    print(target_emb, target_emb.shape)
    model['resnet.positional_embedding'] = target_emb
    torch.save(model, target_path)


if __name__ == '__main__':
    ckpt = 'hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/clue/abvt_b32_tnb_fxtau_1e3_05/model_state_epoch_226687.th'
    target_path = 'model_epoch_226687_repos72.th'
    reconstruct_pos(ckpt, target_path)
