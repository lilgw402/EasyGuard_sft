import torch
import torch.nn as nn
import torch.nn.functional as F

from titan.models.image.resnet.resnet import resnet50
from titan.models.lang.bert_chinese_cm import bert_chinese
from titan.models.fusion.fc_fusion import fc_fusion


# this is the VLN used for CM tasks
class resnet_bert_fc_cm(nn.Module):
    def __init__(self, 
                has_multi_head=True,
                has_binary_head=True,
                has_embedding_head=True, 
                img_embedding=2048,
                lang_embedding=768,
                fc_channels=256,
                num_classes=19,
                ):
        super(resnet_bert_fc_cm, self).__init__()

        self.has_multi_head = has_multi_head
        self.has_binary_head = has_binary_head
        self.has_embedding_head =has_embedding_head

        # define image model (SWIN transformer)
        self.image_net= resnet50(features_only=True, pretrained=False)

        # define lauguage model (BERT), keep the default configs for pre-trained weights.
        self.lang_net=bert_chinese(
            linguistic_type='text',
            has_title_embedding=False, #keep these default values
            has_ocr_embedding=False,
            has_asr_embedding=False,
            has_merge_embedding=False,
            has_title_text=True,
            has_ocr_text=True,
            has_asr_text=False,
            has_merge_text=True,
            mlm_prob=0,
            remain_lm_hidden=False,
            fusion_type='fc_concat',
            out_channels=-1,
            lock_embedding_head=False)

        # define fusion
        self.fusion_net=fc_fusion(
            img_embedding=img_embedding, 
            lang_embedding=lang_embedding,
            fc_channels=fc_channels,
            num_classes=num_classes,
            has_multi_head=self.has_multi_head,
            has_binary_head=self.has_binary_head,
            has_embedding_head=self.has_embedding_head)

        # define criterion

    def forward(self, data_batch):
        if isinstance(data_batch, dict):
            x = data_batch['data']
            targets = data_batch.get('targets')
            meta = data_batch.get('meta')
        else:  # Deploy mode: data_batch is a tensor rather a dict contained with 'data' and 'targets'.
            x, targets, meta = data_batch, None, None


        # get language feature
        feature_hidden_states, feature_linguistic=self.lang_net(data_batch)

        # get image feature
        image_feat = self.image_net(x=x)[3]
        _,c,w,h=image_feat.shape
        image_feat=image_feat.reshape(-1,8,c,w,h).permute(0,2,1,3,4)
        image_feat=torch.mean(image_feat,dim=[2,3,4])

        # get fused feature, return a list of predictions e.g., [fc,emb,img,vid]
        out=self.fusion_net(feat_im=image_feat,feat_lang=feature_linguistic[0])


        # process targets
        # TODO: refine this logic
        if not self.has_binary_head:
            targets = [targets[0], targets[0], targets[0]]
        else:
            binary_targets = (targets[0] >= 1).long()
            targets = [targets[0], binary_targets, binary_targets]
        if self.has_embedding_head:
            targets.insert(1, targets[0])

        return out, targets,meta