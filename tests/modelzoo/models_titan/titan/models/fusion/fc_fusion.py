import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from time import sleep
from titan.models.module.head import MyFCHead, MyClsHead


class fc_fusion(nn.Module):
    def __init__(self, 
                img_embedding=0, 
                vid_embedding=0, 
                lang_embedding=0,
                fc_channels=256,
                num_classes=19,
                has_multi_head=True,
                has_binary_head=False,
                has_embedding_head=True,
                dp=0):
        super(fc_fusion, self).__init__()
        self.fc_channels = fc_channels
        self.num_classes = num_classes
        self.has_multi_head = has_multi_head
        self.has_binary_head = has_multi_head
        self.has_embedding_head = has_multi_head

        self.img_embedding=img_embedding
        self.vid_embedding=vid_embedding
        self.lang_embedding=lang_embedding
        self.mm_embedding=img_embedding+vid_embedding+lang_embedding
        self.dp=dp

        # image head
        if img_embedding:
            self.fc_img = MyFCHead(self.img_embedding, self.fc_channels)
            img_out_channels = self.fc_img.out_channels
            if not self.has_binary_head:
                self.cls_img = MyClsHead(img_out_channels, self.num_classes,dp_ratio=dp)
            else:
                self.cls_img = MyClsHead(img_out_channels, 2,dp_ratio=dp)
        
        # video head
        if vid_embedding:
            self.fc_vid = MyFCHead(self.vid_embedding, self.fc_channels)
            vid_out_channels = self.fc_vid.out_channels
            if not self.has_binary_head:
                self.cls_vid = MyClsHead(vid_out_channels, self.num_classes,dp_ratio=dp)
            else:
                self.cls_vid = MyClsHead(vid_out_channels, 2,dp_ratio=dp)
        
        # language head
        if lang_embedding:
            self.fc_lang = MyFCHead(self.lang_embedding, self.fc_channels)
            lang_out_channels = self.fc_lang.out_channels
            if not self.has_binary_head:
                self.cls_lang = MyClsHead(lang_out_channels, self.num_classes,dp_ratio=dp)
            else:
                self.cls_lang = MyClsHead(lang_out_channels, 2,dp_ratio=dp)

        # concat head 
        if self.has_embedding_head:
            # 128 is embedding dim
            self.fc_e = MyFCHead(self.mm_embedding, 128)
            self.cls_e = MyClsHead(128, self.num_classes,dp_ratio=dp)

            self.fc = MyFCHead(self.mm_embedding, self.fc_channels)
            self.out_channels = self.fc.out_channels
            self.cls = MyClsHead(self.out_channels, self.num_classes,dp_ratio=dp)
        else:
            self.fc = MyFCHead(self.mm_embedding, self.fc_channels)
            self.out_channels = self.fc.out_channels
            self.cls = MyClsHead(self.out_channels, self.num_classes,dp_ratio=dp)
    

    def forward(self, feat_im=None, feat_vid=None, feat_lang=None):
        to_cat=[]
        if feat_im is not None:
            to_cat.append(feat_im)
        if feat_vid is not None: 
            to_cat.append(feat_vid)
        if feat_lang is not None:
            to_cat.append(feat_lang)

        feat_cat=torch.cat(to_cat,dim=1)

        # embedding out and fc out
        if self.has_embedding_head:
            embedding_feature = self.fc_e(feat_cat)
            embedding_feature = self.cls_e(embedding_feature)
            out = self.fc(feat_cat)
            out = self.cls(out)
            out.extend(embedding_feature)
        else:
            out = self.fc(feat_cat)
            out = self.cls(out)

        # modality out
        if feat_im is not None:
            img_feature = self.fc_img(feat_im)
            image_feature = self.cls_img(img_feature)
            out.extend(image_feature)

        if feat_vid is not None:
            vid_feature = self.fc_vid(feat_vid)
            video_feature = self.cls_vid(vid_feature)
            out.extend(video_feature)

        if feat_lang is not None:
            lang_feature = self.fc_lang(feat_lang)
            language_feature = self.cls_lang(lang_feature)
            out.extend(language_feature)
        return out

