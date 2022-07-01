from typing import List

from fex import _logger as logger

try:
    from text_tokenizer import BpeTokenizer
except Exception as e:
    logger.warning(f'text_tokenizer is not installed: {e}. you can install it with: pip3 install http://d.scm.byted.org/api/v2/download/ceph:nlp.tokenizer.py_1.0.0.104.tar.gz')


class WordpieceTokenizer:
    """Break words into subwords."""

    def __init__(self, vocab: str, unk_token: str = "[UNK]", wordpiece_type: str = "bert",
                 max_input_chars_per_word: int = 200,
                 max_tokens_per_input: int = -1,
                 lower_case: bool = False,
                 unk_kept: bool = False,
                 ) -> None:

        self.bpe_tokenizer: BpeTokenizer = BpeTokenizer(vocab,
                                                        clean_text=False,
                                                        lower_case=lower_case,
                                                        split_punc=False,
                                                        max_input_chars_per_word=max_input_chars_per_word,
                                                        unk_kept=unk_kept,
                                                        wordpiece_type=wordpiece_type,
                                                        unk_token=unk_token)
        self.max_tokens_per_input: int = max_tokens_per_input

    def __call__(self, texts: List[str]) -> List[str]:
        output = []
        output_len = 0
        for term in texts:
            if self.max_tokens_per_input > 0 and output_len > self.max_tokens_per_input:
                break
            tokens = self.bpe_tokenizer(term)
            output.extend(tokens)
            output_len += len(tokens)
        return output
