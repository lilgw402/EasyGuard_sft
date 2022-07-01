#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Mask Language Modeling Task Pre-Processing Pipeline'''

from typing import Dict, Tuple
import warnings
import random

from fex.data import BertTokenizer


class MLMPipe:
    """
    预训练的数据文本处理流。实现了`__call__`，输入是文本字符，输出是可以直接模型需要的输入。
    在这个pipeline里一条龙处理下面几个麻烦的事情：
    1. tokenize
    2. mask的策略
    3. truncate、pad以及indexing

    Notes:
        config.DATASET.MASK_STYLE: ['origin', 'ernie']，第一个是随机mask，第二个是根据实体词mask


    """

    def __init__(self, config, *args, **kwargs):
        self.config = config
        do_lower_case = config.NETWORK.DO_LOWER_CASE if hasattr(config.NETWORK, 'DO_LOWER_CASE') else True
        tokenize_emoji = config.NETWORK.TOKENIZE_EMOJI if hasattr(config.NETWORK, 'TOKENIZE_EMOJI') else True
        greedy_sharp = config.NETWORK.GREEDY_SHARP if hasattr(config.NETWORK, 'GREEDY_SHARP') else True
        self.tokenizer = BertTokenizer(config.NETWORK.VOCAB_FILE,
                                       do_lower_case=do_lower_case,
                                       tokenize_emoji=tokenize_emoji,
                                       greedy_sharp=greedy_sharp)
        self.mask_style = config.DATASET.MASK_STYLE

        # self.vocabs_path = config.NETWORK.VOCAB_MASK
        # self.stop_vocabs_path = config.NETWORK.STOP_VOCAB_MASK
        # if self.vocabs_path != '':
        #     self.vocabs = load_vocab(self.vocabs_path)
        # else:
        #     self.vocabs = set()
        # if self.stop_vocabs_path != '':
        #     self.stop_vocabs = load_vocab(self.stop_vocabs_path)
        # else:
        #     self.stop_vocabs = set()

        self.fields = self.config.DATASET.FIELDS  # fields我们期待他是按重要度排序的，因为truncate的逻辑是从后往前
        self.seq_len = self.config.DATASET.SEQ_LEN

        # 2个任务
        self.with_rel_task = config.NETWORK.WITH_REL_LOSS
        self.with_mlm_task = config.NETWORK.WITH_MLM_LOSS

        self.need_query = config.DATASET.NEED_QUERY
        self.PAD = self.tokenizer.vocab['[PAD]']
        self.debug = kwargs.get('debug', False)

    def __call__(self, example: Dict) -> Dict:
        # Task #1: Caption-Image Relationship Prediction
        # 返回加上nsp任务后需要的东西；文本，相关性判断还要mask任务需要的实体等信息
        relationship_label, text_dict = self.sample_relationship_label(example)
        # Task #2: Masked Language Modeling
        # 返回所有的非重复实体词和限定词，将这些词标定位要高概率mask掉的实体词
        if 'ner_info' in text_dict:
            ner_terms = parse_ner_info(text_dict['ner_info'])
        else:
            ner_terms = []
        tokens, input_ids, segment_ids, mlm_labels = self.tokenize_and_mask(text_dict, relationship_label, high_prob_terms=ner_terms)

        res = {
            'input_ids': input_ids,
            'segment_ids': segment_ids,
            'relationship_label': relationship_label,
            'mlm_labels': mlm_labels}

        if self.need_query:
            query_input_ids, query_segment_ids = self.text_to_tensor(text_dict['query'])
            res.update({
                'query_input_ids': query_input_ids,
                'query_segment_ids': query_segment_ids,
            })

        if self.debug:
            res['doc'] = example
            res['tokens'] = tokens
        return res

    def sample_relationship_label(self, example: Dict) -> Tuple[int, Dict]:
        """
        relationship label的生成策略
        Return:
            relationship_label (int): 默认为1，在with_rel_task的时候可能会变0
            text_dict (Dict): text 的dict，如果是负例的是，对应指向随机负例的text
        """
        relationship_label = 1
        text_dict = example

        if self.with_rel_task:
            # 这玩意儿是用模型预先算了一遍的分数，这样可以知道是不是相关（算不算半监督？）
            # if 'tspt_qv_score' in example:
            #     tspt_qv_score = float(example['tspt_qv_score'])
            #     #倾向于pos相关，30%不相关，70%相关；
            #     # 整体逻辑是既然相关了，那就让相关发挥作用大些。
            #     if tspt_qv_score > 0.5:
            #         if random.random() < 0.3 and 'neg' in example:
            #             relationship_label = 0
            #             text_dict = example['neg']
            #             if 'random_neg_title_ner' in example:
            #                 ner_title = example['random_neg_title_ner']
            #     #倾向于pos不相关，80%不相关中有70%是neg数据来表示不相关，10%是用pos数据表示不相关；20%相关用pos来表示；
            #     # 整体逻辑是既然pos都不相关，那就让不相关的数据发挥作用大些。这个70%和上面的70%正好平衡；
            #     else:
            #         if random.random() < 0.8:
            #             relationship_label = 0
            #             if random.random() < 0.7 and 'neg' in example:
            #                 text_dict = example['neg']
            #                 #这是不相关还要做mask的意思？那为何不用上面的，pos再不相关，也比neg相关？不理解
            #                 if 'random_neg_title_ner' in example:
            #                     ner_title = example['random_neg_title_ner']
            # #如果没用模型预判相关性分数，那么就按照50%的概率整nsp
            # else:
            if random.random() < 0.5 and 'neg' in example:
                relationship_label = 0
                text_dict = example['neg']

        return relationship_label, text_dict

    def tokenize_and_mask(self, text_dict, relationship_label, high_prob_terms=None, low_prob_terms=None):
        """
        生成token，以及做mask
        ner_info 用来做ernie mask用的
        """
        # 1. 做完整的tokenize
        domains, word_infos = self.tokenize_domains(text_dict)

        # 2. assemble and truncate
        tokens, segment_ids, word_infos = self.truncate_and_assemble(
            domains,
            length=self.seq_len - len(domains) - 1,
            skip_empty=True,
            word_infos_domains=word_infos)

        # 3. mask
        if self.with_mlm_task:
            if self.mask_style == 'origin':
                tokens, mlm_labels = self.mask_tokens_wwm(tokens, relationship_label, word_infos)
            elif self.mask_style == 'ernie':
                tokens, mlm_labels = self.mask_tokens_wwm(tokens, relationship_label, word_infos, high_prob_terms)

        # 4. 转id
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return tokens, input_ids, segment_ids, mlm_labels

    def tokenize_domains(self, text_dict):
        """
        做tokenize，保留词的信息，用于在后面mask的时候做全词mask
        """
        domains = []
        word_infos = []
        # 这个干啥？
        offset = 0
        for field in self.fields:
            cur_text = text_dict[field].strip()
            cur_tokens = []
            cur_word_info = []
            if cur_text:
                # cur_word_indo用来保留第一次切词后的信息
                cur_tokens, cur_word_info = self.tokenizer.tokenize_with_ww(cur_text)
                offset += len(cur_text)
                domains.append(cur_tokens)
                word_infos.append(cur_word_info)
            else:
                domains.append(cur_tokens)
                word_infos.append(cur_word_info)
        return domains, word_infos

    def text_to_tensor(self, text):
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[:self.seq_len - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        return input_ids, segment_ids

    def mask_tokens_wwm(self, tokens, relationship_label, word_infos, high_prob_terms=None, low_prob_terms=None,
                        masked_lm_prob=0.15, high_lm_prob=0.3, low_lm_prob=0.05,
                        max_mask_rate=0.5,
                        max_predictions_per_seq=16):
        """
        mask一些token，用whole word mask
        """
        candidate_info = []
        cur_start = 0
        cur_length = 0
        symbol_cnt = 0
        output_tokens = list(tokens)
        output_labels = [-1] * len(tokens)
        if relationship_label == 1:
            # 先确立一些可能被mask的候选，会根据 word_infos 来做全词进入候选
            # 做了个颠倒，UNK的不要被mask掉
            for (i, token) in enumerate(tokens):
                if token in ['[CLS]', '[UNK]', '[SEP]', '[PAD]']:
                    symbol_cnt += 1
                    continue
                elif word_infos[i] != word_infos[i - 1]:
                    if cur_length > 0:
                        candidate_info.append([cur_start, cur_length])
                    cur_start = i
                    cur_length = 1
                else:
                    cur_length += 1
            if cur_length > 0:
                candidate_info.append([cur_start, cur_length])
            # 建了一个候选信息表，是可能会被mask掉的词汇
            random.shuffle(candidate_info)
            # 每句话中最多mask16或者全部的20%，那个少mask哪个
            num_to_predict = min(max_predictions_per_seq, max(
                1, int(round((len(tokens) - symbol_cnt) * max_mask_rate))))

            slots = 0
            has_mask = False

            # 遍历所有候选，都按15%的概率进行mask
            for [start, length] in candidate_info:
                if slots >= num_to_predict:  # 如果够了，就break
                    break
                if slots + length > num_to_predict:  # 如果当前word太长，就continue
                    continue

                prob = random.random()
                # 恢复到原来的term
                recover_term = self.tokenizer.recover(tokens[start:start + length])
                if high_prob_terms and recover_term in high_prob_terms:
                    gate = high_lm_prob
                elif low_prob_terms and recover_term in low_prob_terms:
                    gate = low_lm_prob
                else:
                    gate = masked_lm_prob
                # mask token with x% probability, x=15% by default
                if prob < gate:
                    has_mask = True
                    prob /= gate
                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        masked_tokens = ["[MASK]"] * length
                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        masked_tokens = []
                        for i in range(length):
                            masked_tokens.append(random.choice(list(self.tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
                    else:
                        masked_tokens = tokens[start: start + length]

                    slots += length

                    for i in range(start, start + length):
                        output_tokens[i] = masked_tokens[i - start]
                        try:
                            output_labels[i] = self.tokenizer.vocab[tokens[i]]
                        except KeyError:
                            # For unknown words (should not occur with BPE vocab)
                            output_labels[i] = self.tokenizer.vocab["[UNK]"]
                            warnings.warn("Cannot find sub_token in vocab. Using [UNK] insetad")

            # 最后有一个兜底，如果一个term都没mask掉，还是随机mask一个
            if not has_mask and len(tokens) > 4:
                mask_idx = random.choice(list(range(len(tokens))))
                if tokens[mask_idx] not in ['[CLS]', '[UNK]', '[SEP]', '[PAD]']:
                    prob = random.random()
                    if prob < 0.8:
                        output_tokens[mask_idx] = '[MASK]'
                    elif prob < 0.9:
                        output_tokens[mask_idx] = random.choice(list(self.tokenizer.vocab.keys()))
                    output_labels[mask_idx] = self.tokenizer.vocab[tokens[mask_idx]]

        return output_tokens, output_labels

    @staticmethod
    def truncate_and_assemble(domains, length, skip_empty=True, word_infos_domains=None):
        """
        对所有domain的文本，做合并，以及truncate
        截断策略是，保证越前面的域保留越多
        """
        # [CLS] query [SEP] title [SEP] username [SEP] ... [SEP]
        tokens = ['[CLS]']
        segment_ids = [0]
        word_infos = [-1]
        for i, (domain, word_info) in enumerate(zip(domains, word_infos_domains)):
            if len(domain) == 0 and skip_empty:
                continue
            tokens.extend(domain)
            word_infos.extend(word_info)
            tokens.append('[SEP]')
            segment_ids.extend([i] * (len(domain) + 1))  # 从0开始，domain递增
            word_infos.append(-1)
            if len(tokens) > length - 1:
                # 长度超出了就截断，然后返回
                tokens = tokens[:length - 1]
                segment_ids = segment_ids[:length - 1]
                word_infos = word_infos[:length - 1]
                tokens.append('[SEP]')
                segment_ids.append(i)
                word_infos.append(-1)
                break

        assert len(tokens) == len(segment_ids) == len(word_infos)
        return tokens, segment_ids, word_infos


def parse_ner_info(ner_info):
    nz_terms = set()
    for term in ner_info.strip().split():
        if '#B-NZ' in term or '#I-NZ' in term or '#B-PER' in term or '#I-PER' in term:
            term = term.replace('#B-NZ', '').replace('#I-NZ', '').replace('#B-PER', '').replace('#I-PER', '')
            if term not in ['？', '！', '。', '，', '#', '视频', '助手', '抖音']:  # TODO:
                nz_terms.add(term)
    return list(nz_terms)
