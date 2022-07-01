from typing import List, Dict, Tuple, Any
import os
# import unicodedata

import torch
from torch import nn
try:
    import textops
    from textops.pytorch import load_vocab as textops_load_vocab
    from textops.pytorch.nlp_lib_cut_op import NlpLibCutter
    textops.torch_load_op_library()
except:
    print("No textops found, trace will use textops!")


def sum_2d_list_length(lists: List[List[str]]):
    s: int = 0
    for l in lists:
        s += len(l)
    return s


def tail_first_truncate(domains: List[List[str]], max_length: int):
    domain_size = len(domains)
    new_domains: List[List[str]] = domains
    while sum_2d_list_length(new_domains) > max_length:
        # print(rlengths)
        for i in range(domain_size - 1, -1, -1):
            if len(new_domains[i]) >= 1:
                new_domains[i].pop()
                break
    return new_domains


def clip_pad_1d(tensor: torch.Tensor, pad_length: int, pad: int = 0):
    if tensor.shape[0] == pad_length:
        return tensor
    else:
        tensor_ret = torch.full((pad_length, ), pad,
                                dtype=tensor.dtype, device=tensor.device)
        tensor_ret[:min(tensor.shape[0], pad_length)
                   ] = tensor[:min(tensor.shape[0], pad_length)]
        return tensor_ret


def clip_pad_2d(tensor: torch.Tensor, pad_shape: List[int], pad: int = 0):
    if tensor.shape[0] == pad_shape[0]:
        return tensor
    else:
        tensor_ret = torch.full(
            pad_shape, pad, dtype=tensor.dtype, device=tensor.device)
        tensor_ret[:min(tensor.shape[0], pad_shape[0]), :min(tensor.shape[1], pad_shape[1])] \
            = tensor[:min(tensor.shape[0], pad_shape[0]), :min(tensor.shape[1], pad_shape[1])]
        return tensor_ret


def make_mask_1d(valid_length: int, max_length: int) -> torch.Tensor:
    new_ten = torch.zeros(max_length, dtype=torch.int64)
    new_ten[:min(valid_length, max_length)] = torch.ones(
        min(valid_length, max_length))
    return new_ten.to(dtype=torch.bool)


def load_vocab(vocab_file):
    """
    Loads a vocabulary file into a dictionary.
    For initialization, not in the tracing process.
    """
    vocab: Dict[str, int] = {}
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


class TableLookup(nn.Module):
    def __init__(self, vocab_file):
        super().__init__()
        self.vocab: Dict[str, int] = load_vocab(vocab_file)

    def forward(self, tokens: List[str]):
        ids: List[int] = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids


class MultiDomainConcator(nn.Module):
    def __init__(self, vocab_file: str):
        super().__init__()
        self.table_looup = torch.jit.script(TableLookup(vocab_file))

    def forward(self, query_tok: List[str], domains: List[List[str]], skip_empty: bool = True, skip_query: bool = False, skip_domains: bool = False):
        inputs: List[str] = ['[CLS]']
        segment_ids = [0]

        if not skip_query:
            inputs.extend(query_tok)
            inputs.append('[SEP]')
            segment_ids = [0] * len(inputs)

        if not skip_domains:
            for i, domain in enumerate(domains):
                if len(domain) == 0 and skip_empty:
                    continue
                inputs.extend(domain)
                inputs.append('[SEP]')
                segment_ids.extend([i + 1] * (len(domain) + 1))
        assert len(inputs) == len(segment_ids)
        input_ids = self.table_looup(inputs)
        return input_ids, segment_ids


class PyBertTokenizer(nn.Module):
    vocab: Dict[str, int]

    def __init__(self,
                 vocab_file,
                 do_lower_case: bool = True,
                 do_word_piece: bool = True,
                 wordpiece_type: str = "no-bert",
                 unk: str = "[UNK]"):
        super().__init__()
        self.vocab: Dict[str, int] = load_vocab(vocab_file)
        self.do_lower_case: bool = do_lower_case
        self.do_word_piece: bool = do_word_piece
        self.wordpiece_type: str = wordpiece_type
        self.unk_token: str = unk

    @torch.jit.export
    def __tokenize_(self, token: str) -> List[str]:
        MAX_CHAR_WORD_LEN = 100

        outputs: List[str] = []
        if self.do_lower_case:
            token = token.lower()
        if len(token) > MAX_CHAR_WORD_LEN:
            outputs.append(self.unk_token)

        is_bad: bool = False
        start: int = 0
        sub_tokens: List[str] = []

        while start < len(token):
            end: int = len(token)
            cur_substr: str = ""
            sub_str: str = ""
            while start < end:
                sub_str = token[start: end]
                if start > 0 and self.do_word_piece and self.wordpiece_type == "bert":
                    sub_str = "##" + sub_str
                if sub_str in self.vocab:
                    cur_substr = sub_str
                    break
                end -= 1
            if cur_substr == "":
                is_bad = True
                break
            sub_tokens.append(cur_substr)
            start = end
        if is_bad:
            outputs.append(self.unk_token)
        else:
            outputs.extend(sub_tokens)
        return outputs

    def forward(self, sentence: str) -> List[List[str]]:
        tokens: List[str] = sentence.split()
        outputs: List[List[str]] = []
        for tok in tokens:
            toks = self.__tokenize_(tok)
            outputs.append(toks)
        return outputs


class ScriptBertTokenizer(nn.Module):
    vocab: Dict[str, int]

    def __init__(self, vocab_file):
        super().__init__()
        vocab = textops_load_vocab(vocab_file)
        self.tokenizer = torch.classes.nlp_bert_tokenize.BertTokenizer(
            vocab, True, False, "bert", "[UNK]", 0)

    def forward(self, text: str) -> List[List[str]]:
        tokens: List[str] = text.split(" ")
        return self.tokenizer.tokenize(tokens)

    # @torch.jit.export
    # def clean_text(self, text: str) -> str:
    #     output = []
    #     for char in text:
    #         cp = ord(char)
    #         if cp == 0 or cp == 0xfffd or self._is_control(char):
    #             continue
    #         if self._is_whitespace(char):
    #             output.append(" ")
    #         else:
    #             output.append(char)
    #     return "".join(output)

    # @torch.jit.export
    # def _is_control(self, char: str):
    #     """Checks whether `chars` is a control character."""
    #     # These are technically control characters but we count them as whitespace
    #     # characters.
    #     if char == "\t" or char == "\n" or char == "\r":
    #         return False
    #     cat = unicodedata.category(char)
    #     if cat.startswith("C"):
    #         return True
    #     return False

    # @torch.jit.export
    # def _is_whitespace(self, char: str):
    #     """Checks whether `chars` is a whitespace character."""
    #     # \t, \n, and \r are technically contorl characters but we treat them
    #     # as whitespace since they are generally considered as such.
    #     if char == " " or char == "\t" or char == "\n" or char == "\r":
    #         return True
    #     cat = unicodedata.category(char)
    #     if cat == "Zs":
    #         return True
    #     return False


if __name__ == "__main__":
    bert_tokenizer = ScriptBertTokenizer(
        vocab_file="/mnt/nlp-lq/bert/vocabs/fine_fix.txt")
    # tokens = bert_tokenizer.forward(
    #     "怎么给小朋友讲汤普森穿过森林", need_cut=True)  # 怎么 给 小朋友 讲 汤普森 穿过 森林
    # print(tokens)
