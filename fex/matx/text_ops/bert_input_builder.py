# -*- coding: utf-8 -*-
from typing import List, Dict, Tuple
try:
    import matx
except Exception:
    print("No Matx or Matx_text found, please check! ")


def clip_pad_list_ids(input_ids: List[int], max_len: int,
                      pad_token_id: int) -> List[int]:
    raw_data_len = len(input_ids)
    if raw_data_len > max_len:
        return input_ids[:max_len]
    else:
        pad_data_ids = matx.List([pad_token_id]) * (max_len - raw_data_len)
        input_ids.extend(pad_data_ids)
    return input_ids


def make_doc_mask(pad_input_ids: List[int], max_len: int,
                  pad_id: int) -> List[int]:
    doc_masks = matx.List()
    doc_masks.reserve(max_len)
    for i in range(max_len):
        doc_masks.append((pad_input_ids[i] != pad_id))
    return doc_masks


class Vocabulary:
    def __init__(self, vocab_file: str, unk_token: str) -> None:
        self.vocab: Dict[str, int] = matx.Dict()
        self.unk: str = unk_token

        fr = open(vocab_file)
        idx = 0
        for token in fr:
            token = token.strip()
            self.vocab[token] = idx
            idx += 1
        fr.close()

    def lookup(self, key: str) -> int:
        if key in self.vocab:
            return self.vocab[key]
        else:
            return self.vocab[self.unk]


class Encoder():
    def __init__(self,
                 vocab_file: str,
                 pad_token: str = '[PAD]',
                 unk_token: str = '[UNK]') -> None:
        self.vocab: Vocabulary = Vocabulary(vocab_file, unk_token)
        self.pad_id: int = self.vocab.lookup(pad_token)

    def __call__(self, input_token: List[str], segment_ids: List[int],
                 max_seq_len: int) -> Tuple[List[int], List[int], List[int]]:
        pad_input_ids: List[int] = self.convert_to_ids_and_pad(
            input_token, max_seq_len, self.pad_id)
        pad_segment_ids: List[int] = clip_pad_list_ids(segment_ids,
                                                       max_seq_len, 0)
        pad_mask_ids: List[int] = make_doc_mask(pad_input_ids, max_seq_len,
                                                self.pad_id)
        return pad_input_ids, pad_segment_ids, pad_mask_ids

    def convert_to_ids_and_pad(self, tokens: List[str], seq_len: int,
                               padding_value: int) -> List[int]:
        input_ids = matx.List()
        input_ids.reserve(seq_len)
        for i in range(min(seq_len, len(tokens))):
            input_ids.append(self.vocab.lookup(tokens[i]))

        left = seq_len - len(input_ids)
        for i in range(left):
            input_ids.append(padding_value)

        return input_ids


class BertInputsBuilder:
    def __init__(self,
                 max_seq_len: int,
                 vocab_file: str,
                 pad_token: str = '[PAD]',
                 unk_token: str = '[UNK]',
                 task_manager: object = None) -> None:
        self.max_seq_len: int = max_seq_len
        self.encoder: Encoder = Encoder(vocab_file, pad_token, unk_token)
        self.task_manager: object = task_manager

    def __call__(
        self,
        batch_inputs_tokens: List[List[str]],
        batch_segment_ids: List[List[int]],
        padding_even: bool = False
    ) -> Tuple[matx.NDArray, matx.NDArray, matx.NDArray]:
        """ 根据 max_seq_len，将batch_inputs_tokens和batch_segment_ids进行padding，生成mask_ids .

        ps: ptx要求seq长度为偶数，因为doc要补上9的，所以需要补刀奇数，而query则需要补到偶数
        Args:
            batch_inputs_tokens (List[List[str]]): batch input tokens
            batch_segment_ids (List[List[int]]): batch segment ids

        Returns:
            Tuple[matx.NDArray, matx.NDArray, matx.NDArray]: 返回 input_ids_tensor, segment_ids_tensor和mask_ids_tensor .
        """
        batch_size: int = len(batch_inputs_tokens)

        batch_pad_inputs_ids = matx.List()
        batch_pad_segment_ids = matx.List()
        batch_mask_ids = matx.List()

        batch_pad_inputs_ids.reserve(batch_size)
        batch_pad_segment_ids.reserve(batch_size)
        batch_mask_ids.reserve(batch_size)

        max_seq_len = 1
        for x in batch_inputs_tokens:
            max_seq_len = max(max_seq_len, len(x))

        if padding_even and max_seq_len % 2 == int(padding_even):
            max_seq_len += 1

        if self.task_manager is not None:
            futures = []
            for index in range(batch_size):
                futures.append(self.task_manager.get_thread_pool().Submit(
                    self.encoder, batch_inputs_tokens[index],
                    batch_segment_ids[index], max_seq_len))

            for future in futures:
                pad_input_ids, pad_segment_ids, pad_mask_ids = future.get()
                batch_pad_inputs_ids.append(pad_input_ids)
                batch_pad_segment_ids.append(pad_segment_ids)
                batch_mask_ids.append(pad_mask_ids)
        else:
            for index in range(batch_size):
                pad_input_ids, pad_segment_ids, pad_mask_ids = self.encoder(
                    batch_inputs_tokens[index], batch_segment_ids[index],
                    max_seq_len)
                batch_pad_inputs_ids.append(pad_input_ids)
                batch_pad_segment_ids.append(pad_segment_ids)
                batch_mask_ids.append(pad_mask_ids)

        input_tensor = matx.NDArray(batch_pad_inputs_ids, [], "int64")
        segment_tensor = matx.NDArray(batch_pad_segment_ids, [], "int64")
        mask_tensor = matx.NDArray(batch_mask_ids, [], "int64")
        return input_tensor, segment_tensor, mask_tensor


class BertQueryStackOnDocsInputsBuilder:
    def __init__(self,
                 max_seq_len: int,
                 vocab_file: str,
                 pad_token: str = '[PAD]',
                 unk_token: str = '[UNK]') -> None:
        self.max_seq_len: int = max_seq_len
        self.vocab: Vocabulary = Vocabulary(vocab_file, unk_token)
        self.pad_id: int = self.vocab.lookup(pad_token)

    def __call__(
        self, batch_query_inputs_tokens: List[List[str]],
        batch_query_segment_ids: List[List[int]],
        batch_docs_inputs_tokens: List[List[str]],
        batch_docs_segment_ids: List[List[int]]
    ) -> Tuple[matx.NDArray, matx.NDArray, matx.NDArray]:
        """ 将query堆叠在doc上，整体生成bert input，最终batch为 doc_batch_size + query_batch_size;
        同时将 batch_query_inputs_tokens 和 batch_doc_inputs_tokens 根据 batch 内部的 max_len 做 padding.

        Args:
            self ([type]): [description]

        Returns:
            Tuple[matx.NDArray, matx.NDArray, matx.NDArray]: [description]
        """
        query_batch_size: int = len(batch_query_inputs_tokens)
        doc_batch_size: int = len(batch_docs_inputs_tokens)
        total_batch_size: int = query_batch_size + doc_batch_size

        total_input_ids = matx.List()
        total_input_ids.reserve(total_batch_size)
        total_segment_ids = matx.List()
        total_input_ids.reserve(total_batch_size)
        total_mask_ids = matx.List()
        total_mask_ids.reserve(total_batch_size)

        max_seq_len: int = self.calcualte_inner_max_len(
            batch_query_inputs_tokens, batch_docs_inputs_tokens)

        for index in range(query_batch_size):
            query_input_ids = self.convert_to_ids(
                batch_query_inputs_tokens[index])
            query_segment_ids = batch_query_segment_ids[index]

            query_pad_input_ids = clip_pad_list_ids(query_input_ids,
                                                    max_seq_len, self.pad_id)
            query_pad_segment_ids = clip_pad_list_ids(query_segment_ids,
                                                      max_seq_len, 0)

            total_input_ids.append(query_pad_input_ids)
            total_segment_ids.append(query_pad_segment_ids)
            total_mask_ids.append(
                make_doc_mask(query_pad_input_ids, max_seq_len, self.pad_id))

        for index in range(doc_batch_size):
            doc_input_ids = self.convert_to_ids(
                batch_docs_inputs_tokens[index])
            doc_segment_ids = batch_docs_segment_ids[index]

            doc_pad_input_ids = clip_pad_list_ids(doc_input_ids, max_seq_len,
                                                  self.pad_id)
            doc_pad_segment_ids = clip_pad_list_ids(doc_segment_ids,
                                                    max_seq_len, 0)

            total_input_ids.append(doc_pad_input_ids)
            total_segment_ids.append(doc_pad_segment_ids)
            total_mask_ids.append(
                make_doc_mask(doc_pad_input_ids, max_seq_len, self.pad_id))

        input_tensor = matx.NDArray(total_input_ids, [], "int64")
        segment_tensor = matx.NDArray(total_segment_ids, [], "int64")
        mask_tensor = matx.NDArray(total_mask_ids, [], "int64")
        return input_tensor, segment_tensor, mask_tensor

    def calcualte_inner_max_len(
            self, batch_query_inputs_tokens: List[List[str]],
            batch_docs_inputs_tokens: List[List[str]]) -> int:
        # cacluate inner max_len in queries and docs
        query_max_len = 0
        doc_max_len = 0
        query_batch_size: int = len(batch_query_inputs_tokens)
        doc_batch_size: int = len(batch_docs_inputs_tokens)

        for index in range(query_batch_size):
            query_tokens_len = len(batch_query_inputs_tokens[index])
            if query_tokens_len > query_max_len:
                query_max_len = query_tokens_len

        for index in range(doc_batch_size):
            doc_tokens_len = len(batch_docs_inputs_tokens[index])
            if doc_tokens_len > doc_max_len:
                doc_max_len = doc_tokens_len

        max_seq_len = max(query_max_len, doc_max_len)
        if max_seq_len > self.max_seq_len:
            return self.max_seq_len
        return max_seq_len

    def convert_to_ids(self, tokens: List[str]) -> List[int]:
        input_ids = matx.List()
        input_ids.reserve(len(tokens))
        for token in tokens:
            input_ids.append(self.vocab.lookup(token))
        return input_ids
