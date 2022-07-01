# -*- coding: utf-8 -*-

from typing import List, Tuple
try:
    import matx
except Exception:
    print("No Matx or Matx_text found, please check! ")


class _MultiDomainConcatBuilder:
    def __init__(self, max_seq_len: int, skip_empty: bool = True) -> None:
        self.max_seq_len: int = max_seq_len
        self.skip_empty: bool = skip_empty

    def __call__(
            self, domains: List[List[str]], domain_segment: List[int],
            every_domain_max_lens: List[int]) -> Tuple[List[str], List[int]]:
        inputs = matx.List(['[CLS]'])
        segment_ids = matx.List([0])
        inputs.reserve(self.max_seq_len)
        segment_ids.reserve(self.max_seq_len)

        domain_size = len(domains)
        pre_domain_segments_id = domain_segment[0]
        max_tokens_len = self.max_seq_len - 1
        inputs_size = 1  # [CLS]

        for domain_index in range(domain_size):
            domain_max_len: int = every_domain_max_lens[domain_index]
            domain_tokens = domains[domain_index]
            len_domain_tokens = len(domain_tokens)
            if domain_max_len > 0 and len_domain_tokens > domain_max_len:
                domain_tokens = domain_tokens[:domain_max_len]
                len_domain_tokens = domain_max_len
            if len_domain_tokens == 0 and self.skip_empty:
                continue

            domain_segment_id = domain_segment[domain_index]
            # 如果两个域的segment_id相同，则不会在之间加上 [SEP] token
            if domain_index > 0 and domain_segment_id != pre_domain_segments_id:
                inputs.append('[SEP]')
                segment_ids.append(pre_domain_segments_id)
                inputs_size += 1
            # append的效率比extend要高，这和matx实现有关系
            cur_max_len = max_tokens_len - inputs_size
            if cur_max_len > len_domain_tokens:
                cur_max_len = len_domain_tokens
            for ti in range(cur_max_len):
                inputs.append(domain_tokens[ti])
                segment_ids.append(domain_segment_id)
            inputs_size += cur_max_len
            pre_domain_segments_id = domain_segment_id
            if inputs_size == max_tokens_len:
                break
        if domain_size > 0:
            inputs.append('[SEP]')
            segment_ids.append(pre_domain_segments_id)

        return inputs, segment_ids


class MultiDomainConcatBuilder:
    def __init__(self,
                 max_seq_len: int,
                 skip_empty: bool = True,
                 task_manager: object = None) -> None:
        self.single_call_op: _MultiDomainConcatBuilder = _MultiDomainConcatBuilder(
            max_seq_len, skip_empty)
        self.task_manager: object = task_manager

    def __call__(
        self, domains: List[List[List[str]]], domain_segments: List[int],
        every_domain_max_lens: List[int]
    ) -> Tuple[List[List[str]], List[List[int]]]:
        """ 使用matx实现多域拼接。
        domains: 包含了所有的域，每个域都是batch输入
        domain_segments: 每个域对应的segment_id
        every_domain_max_lens: 每个域的最大长度，超过最大长度会截断(默认所有域都不做限制)

        Returns:
            Tuple[List[List[str]], List[List[int]]]： 返回 batch_input_tokens, batch_segment_ids
        """

        batch_size = len(domains[0])
        batch_inputs = matx.List()
        batch_segment_ids = matx.List()

        batch_inputs.reserve(batch_size)
        batch_segment_ids.reserve(batch_size)

        if self.task_manager is not None:
            futures = []
            for i in range(batch_size):
                tmp_domains = []
                for x in domains:
                    tmp_domains.append(x[i])

                futures.append(self.task_manager.get_thread_pool().Submit(
                    self.single_call_op, tmp_domains,
                    domain_segments, every_domain_max_lens))

            for future in futures:
                inputs, segment_ids = future.get()
                batch_inputs.append(inputs)
                batch_segment_ids.append(segment_ids)
        else:
            for i in range(batch_size):
                tmp_domains = []
                for x in domains:
                    tmp_domains.append(x[i])

                inputs, segment_ids = self.single_call_op(
                    tmp_domains, domain_segments,
                    every_domain_max_lens)
                batch_inputs.append(inputs)
                batch_segment_ids.append(segment_ids)

        return batch_inputs, batch_segment_ids
