# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import torch
from fex.matx.text_ops import BertTokenizer, MultiDomainConcatBuilder, \
    BertInputsBuilder, BertQueryStackOnDocsInputsBuilder, EmbedProcess

try:
    import matx
    import matx_text
    import matx_pytorch
except Exception:
    print("No Matx or Matx_text found, Con't use MatxTextBasePipe ! ")


class MatxTextBasePipe(ABC):
    def __init__(self,
                 vocab_file: str,
                 max_seq_len: int,
                 image_token_num: int = 8,
                 image_feature_dim: int = 128,
                 do_lower_case: bool = True,
                 tokenize_emoji: bool = False,
                 greedy_sharp: bool = True,
                 to_trace: bool = False):
        self.vocab_file = vocab_file
        self.do_lower_case = do_lower_case
        self.tokenize_emoji = tokenize_emoji
        self.greedy_shape = greedy_sharp
        self.image_token_num = image_token_num
        self.image_feature_dim = image_feature_dim

        word_piece_tokenizer = matx_text.WordPieceTokenizerOp(
            location=self.vocab_file)

        self.matx_bert_tokenizer = matx.script(BertTokenizer)(tokenizer=word_piece_tokenizer,
                                                              do_lower_case=self.do_lower_case,
                                                              tokenize_emoji=self.tokenize_emoji,
                                                              greedy_sharp=self.greedy_shape)

        self.multi_domain_concat_builder = matx.script(
            MultiDomainConcatBuilder)(max_seq_len=max_seq_len)

        self.build_input_builder = matx.script(BertInputsBuilder)(
            max_seq_len=48, vocab_file=self.vocab_file)

        self.build_query_stack_on_docs_input_builder = matx.script(
            BertQueryStackOnDocsInputsBuilder)(max_seq_len=48, vocab_file=self.vocab_file)

        self.embedding_process = EmbedProcess(image_token_num=self.image_token_num,
                                              image_feature_dim=self.image_feature_dim)

        if to_trace:
            self.embedding_process = matx_pytorch.InferenceOp(model=self.embedding_process,
                                                              trace_func=torch.jit.script,
                                                              device=-1)

    @abstractmethod
    def train_process(self):
        """ train_process 将文本str经过 Tokenizer -> MulitiDomainConcat -> ConvertToIdAndPad

        Returns:
            Tuple[List[int], List[int], List[int]]: 返回 input_ids, segment_ids and mask_ids .

        """

        raise NotImplementedError("Please Implement 'train_process' method ! ")

    @abstractmethod
    def trace_process(self):
        """ trace_process 是model上线过程中matx.pipeline.Trace的function.
        将将输入通过 Tokenizer -> MulitiDomainConcat -> ConvertToIdAndPad -> VisionEmbProcess -> PytorchModel -> PostProcess

        Returns: Model或者PostProcess的结果

        """

        raise NotImplementedError("Please Implement 'trace_process' method ! ")
