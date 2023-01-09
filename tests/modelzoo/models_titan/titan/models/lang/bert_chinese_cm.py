import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertModel
from titan.utils.helper import download_weights
from titan.utils.misc import get_lang_checkpoint_file
from titan.utils.tos_helper import TOSHelper

class LinguisticModel(nn.Module):

    def __init__(self, 
                linguistic_type='text',
                has_title_embedding=False,
                has_ocr_embedding = False,
                has_asr_embedding = False,
                has_merge_embedding= False,
                title_embedding_dim=128,
                ocr_embedding_dim =128,
                asr_embedding_dim =128,
                fusion_type='fc_concat',
                out_channels=768,

                ):
        super(LinguisticModel, self).__init__()

        # init toshelper for bertchinese
        tos_helper = TOSHelper(
            'haggs-model-zoo',
            'Z2IXQ4ALLPM1H2O39Y3Q')

        # embedding or text
        self.linguistic_type = linguistic_type

        self.has_title_embedding = has_title_embedding
        self.has_ocr_embedding = has_ocr_embedding
        self.has_asr_embedding = has_asr_embedding
        self.has_merge_embedding = has_merge_embedding

        self.title_embedding_dim = title_embedding_dim
        self.ocr_embedding_dim = title_embedding_dim
        self.asr_embedding_dim = title_embedding_dim

        if self.linguistic_type == 'embedding':

            if self.has_merge_embedding:
                self.embedding_dim = 0
                if self.has_title_embedding:
                    self.embedding_dim += self.title_embedding_dim
                if self.has_title_embedding:
                    self.embedding_dim += self.ocr_embedding_dim
                if self.has_title_embedding:
                    self.embedding_dim += self.asr_embedding_dim
            else:
                # TODO: what if they are different ?
                assert self.title_embedding_dim == self.ocr_embedding_dim == self.asr_embedding_dim
                self.embedding_dim = self.title_embedding_dim
            self.out_channels = self.embedding_dim

        elif self.linguistic_type == 'text':
            local_file=download_weights(
                                'bert_cm',
                                tos_helper=tos_helper,
                                pretrained_version='bert_cm_chinese',
                                pretrained_uri='')
            from  transformers import BertModel
            self.fusion_type = fusion_type
            self.embedding_only = False
            if 'transformer' in self.fusion_type and 'custom_transformer' not in self.fusion_type:
                self.embedding_only = True

            self.len_img_mask = None


            bert_model = BertModel.from_pretrained(local_file)
            if self.embedding_only:
                self.bert_model = bert_model.embeddings
            else:
                self.bert_model = BertModel.from_pretrained(local_file)
            self.out_channels = out_channels
        else:
            raise ValueError

    def forward(self, *args):
        if self.linguistic_type == 'embedding':
            hidden_states, feature_input = None, args[0]
            if self.use_text_cnn:
                s = feature_input.shape
                r = self.num_repeats
                feature_repeat = feature_input.repeat(1, r).reshape((s[0], r, s[1])).permute(0, 2, 1)  # (batch_size, 128, r)
                feature_out = torch.cat([self.pool(F.relu(each_conv(feature_repeat))).squeeze(-1) for each_conv in self.convs], dim=1)
            else:
                feature_out = feature_input
        elif self.linguistic_type == 'text':
            import transformers
            from packaging import version
            if self.embedding_only:
                hidden_states = self.bert_model(input_ids=args[0].long())
                feature_out = args[1]
            elif version.parse(transformers.__version__) >= version.parse('4.0.0'):
                temp = self.bert_model(input_ids=args[0].long(), attention_mask=args[1])
                hidden_states = temp['last_hidden_state']
                feature_out = temp['pooler_output']
                if self.fusion_type == 'custom_transformer':
                    feature_out = args[1]
            else:
                if self.fusion_type == 'custom_transformer':
                    hidden_states, feature_out = self.bert_model(input_ids=args[0].long(), attention_mask=args[1][:,self.len_img_mask:])
                    feature_out = args[1]
                else:
                    hidden_states, feature_out = self.bert_model(input_ids=args[0].long(), attention_mask=args[1])
        else:
            raise ValueError
        return hidden_states, feature_out


class LinguisticHead(nn.Module):

    def __init__(self, 
                
                ):
        super(LinguisticHead, self).__init__()

        tos_helper = TOSHelper(
            'haggs-model-zoo',
            'Z2IXQ4ALLPM1H2O39Y3Q')

        local_file=download_weights(
                                'bert_cm',
                                tos_helper=tos_helper,
                                pretrained_version='bert_cm_chinese',
                                pretrained_uri='')
        import BertModel
        # from transformers import BertConfig
        from transformers.models.bert.modeling_bert import BertLMPredictionHead
        bert_model = BertModel.from_pretrained(local_file)
        self.predictions = BertLMPredictionHead(bert_model.config)
        self.predictions.decoder.weight = bert_model.embeddings.word_embeddings.weight
        self.config = bert_model.config

    def forward(self, hidden_states):
        return self.predictions(hidden_states)




class bert_chinese(nn.Module):

    def __init__(self, 
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
                lock_embedding_head=False):
        super(bert_chinese, self).__init__()

        tos_helper = TOSHelper(
            'haggs-model-zoo',
            'Z2IXQ4ALLPM1H2O39Y3Q')

        # embedding or text
        self.linguistic_type = linguistic_type

        self.has_title_embedding = has_title_embedding
        self.has_ocr_embedding = has_ocr_embedding
        self.has_asr_embedding = has_asr_embedding
        self.has_merge_embedding = has_merge_embedding

        self.has_title_text = has_title_text
        self.has_ocr_text = has_ocr_text
        self.has_asr_text = has_asr_text
        self.has_merge_text = has_merge_text

        self.mlm_enable = mlm_prob > 0
        self.remain_lm_hidden = remain_lm_hidden
        self.mrfr_prob = 0.0
        self.mrfr_enable = self.mrfr_prob > 0
        self.fusion_type = fusion_type
        self.embedding_only = False
        self.share_lm_model = False
        if 'transformer' in self.fusion_type and 'custom_transformer' not in self.fusion_type:
            self.embedding_only = True

        if self.linguistic_type == 'text':
            if self.has_merge_text:
                self.text_prefixes = ['']
            else:
                self.text_prefixes = []
                if self.has_title_text:
                    self.text_prefixes.append('title' + '_')
                if self.has_ocr_text:
                    self.text_prefixes.append('ocr' + '_')
                if self.has_asr_text:
                    self.text_prefixes.append('asr' + '_')
        else:
            pass

        if self.linguistic_type == 'text':
            self.prefixes = self.text_prefixes
        else:
            raise ValueError

        local_file=download_weights(
                                'bert_cm',
                                tos_helper=tos_helper,
                                pretrained_version='bert_cm_chinese',
                                pretrained_uri='')
        self.fusion_type = fusion_type
        self.embedding_only = False
        if 'transformer' in self.fusion_type and 'custom_transformer' not in self.fusion_type:
            self.embedding_only = True

        self.len_img_mask = None

        bert_model = BertModel.from_pretrained(local_file)
        if self.embedding_only:
            self.bert_model = bert_model.embeddings
        else:
            self.bert_model = BertModel.from_pretrained(local_file)
        self.out_channels = out_channels


        self.linguistic_model = nn.ModuleList()

        for i, p in enumerate(self.prefixes):
            self.linguistic_model.append(LinguisticModel())
        
        if lock_embedding_head:
            for param in self.linguistic_model.parameters():
                param.requires_grad = False

        print("********* BERT Initialized **************")


    def forward(self, data_batch):
        feature_linguistic = []
        feature_hidden_states=[]
        masks_list = []

        if self.linguistic_type == 'text':
            for i, p in enumerate(self.text_prefixes):
                input_ids = data_batch['extra'][p + 'input_ids']
                attention_mask = data_batch['extra'][p + 'attention_mask']
                masks_list.append(attention_mask)
                if self.share_lm_model:
                    hidden_states, feature = self.linguistic_model[0](input_ids, attention_mask)
                else:
                    hidden_states,feature = self.linguistic_model[i](input_ids, attention_mask)
                if self.fusion_type in ['fcca','fcsa','fc_concat']:
                    feature_hidden_states.append(hidden_states)
                feature_linguistic.append(feature)
        else:
            raise ValueError

        return feature_hidden_states, feature_linguistic

