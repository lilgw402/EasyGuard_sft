# -*- coding: utf-8 -*-

from typing import List, Tuple
try:
    import matx
except Exception:
    print("No Matx or Matx_text found, please check! ")


class StackQueriesWithDocs:
    def __init__(self) -> None:
        pass

    def __call__(self,
                 batch_query_input_ids: List[List[int]],
                 batch_query_segment_ids: List[List[int]],
                 batch_query_mask_ids: List[List[int]],
                 batch_doc_input_ids: List[List[int]],
                 batch_doc_segment_ids: List[List[int]],
                 batch_doc_mask_ids: List[List[int]]) -> Tuple[matx.NDArray, matx.NDArray, matx.NDArray]:
        """ 将query堆叠在doc上，默认batch_query的batch_size=1，最终总的batch_size+1

        Args:
            batch_query_input_ids (List[List[int]]): query的input_ids，默认batch_size=1
            batch_query_segment_ids (List[List[int]]): query的segment_ids
            batch_query_mask_ids (List[List[int]]): query的mask_ids
            batch_doc_input_ids (List[List[int]]): 所有doc的input_ids
            batch_doc_segment_ids (List[List[int]]): 所有doc的segment_ids
            batch_doc_mask_ids (List[List[int]]): 所有doc的mask_ids

        Returns:
            Tuple[matx.NDArray, matx.NDArray, matx.NDArray]: 返回堆叠后的batch_input_tensor, batch_segment_tensor, batch_mask_tensor.
        """
        batch_size = len(batch_query_input_ids) + 1

        total_input_ids = matx.List()
        total_input_ids.reserve(batch_size)
        total_segment_ids = matx.List()
        total_input_ids.reserve(batch_size)
        total_mask_ids = matx.List()
        total_mask_ids.reserve(batch_size)

        total_input_ids.append(batch_query_input_ids[0])
        total_input_ids.extend(batch_doc_input_ids)

        total_segment_ids.append(batch_query_segment_ids[0])
        total_segment_ids.extend(batch_doc_segment_ids)

        total_mask_ids.append(batch_query_mask_ids[0])
        total_mask_ids.extend(batch_query_mask_ids)

        total_input_tensor = matx.NDArray(total_input_ids, [], "int32")
        total_segment_tensor = matx.NDArray(total_segment_ids, [], "int32")
        total_mask_tensor = matx.NDArray(total_mask_ids, [], "int32")

        return total_input_tensor, total_segment_tensor, total_mask_tensor
