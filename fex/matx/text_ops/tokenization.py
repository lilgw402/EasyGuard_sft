# -*- coding: utf-8 -*-

from typing import List
import unicodedata
try:
    import matx
    from matx import FTList, FTSet, FTDict
except Exception:
    print("No Matx or Matx_text found, please check! ")


class TaskManager:

    def __init__(self, pool_size: int, use_lockfree_pool: bool) -> None:
        self.thread_pool: matx.NativeObject = matx.make_native_object("ThreadPoolExecutor", pool_size, use_lockfree_pool)

    def get_thread_pool(self) -> object:
        return self.thread_pool


class CleanText:

    def __init__(self, do_lower_case: bool, tokenize_emoji: bool, tokenizer: matx.NativeObject, need_clean_text: bool = False) -> None:
        self.do_lower_case: bool = do_lower_case
        self.tokenize_emoji: bool = tokenize_emoji
        self.never_split: FTSet[str] = {"[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"}
        self.tokenizer: matx.NativeObject = tokenizer
        self.punctuation_set: FTSet[int] = {33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 58, 59, 60, 61, 62, 63, 64, 91, 92, 93, 94, 95, 96, 123, 124, 125, 126}

    def is_alpha_or_digit(self, cp: int) -> bool:
        if cp > 0x7a:
            return False

        if (cp >= 0x41 and cp <= 0x5a) or \
           (cp >= 0x61 and cp <= 0x7a) or (cp >= 0x30 and cp <= 0x39):
            return True
        else:
            return False

    def is_emoji_char(self, char_code: int) -> bool:
        if not char_code:
            return False

        if (129750 < char_code) or (174 < char_code < 8205) or (12953 < char_code < 126980) or (127569 < char_code < 127744) or (char_code < 169):
            return False
        return True

    def is_control(self, char: str, cat: str) -> bool:
        """Checks whether `chars` is a control character."""
        # 之前过滤了，不起作用
        #         if char == "\t" or char == "\n" or char == "\r":
        #             return False
        if cat[0] == "C":
            return True
        return False

    def is_punctuation(self, cp: int, cat: str) -> bool:
        # if cat[0] == 'P' or ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        #     (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):

        if cat[0] == 'P' or cp in self.punctuation_set:
            return True
        return False

    def is_mask(self, cp: int, cat: str) -> bool:
        if cat[:2] == "Mn":
            return True
        return False

    def is_whitespace(self, char: str, cat: str) -> bool:
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if cat == "Zs" or char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        return False

    def is_chinese_char(self, cp: int) -> bool:
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or (cp >= 0x3400 and cp <= 0x4DBF)    #
                or (cp >= 0x20000 and cp <= 0x2A6DF)    #
                or (cp >= 0x2A700 and cp <= 0x2B73F)    #
                or (cp >= 0x2B740 and cp <= 0x2B81F)    #
                or (cp >= 0x2B820 and cp <= 0x2CEAF)    #
                or (cp >= 0xF900 and cp <= 0xFAFF) or (cp >= 0x2F800 and cp <= 0x2FA1F)    #
                ):    #
            return True

        return False

    def clean_text(self, text: str) -> List[str]:
        # char级别的处理
        text = unicodedata.normalize("NFD", text)
        if self.do_lower_case:
            text = text.lower()
        tokens: List[str] = text.split(' ')
        output_tokens = matx.List()
        output_tokens.reserve(2 * len(tokens))
        for token in tokens:
            if token in self.never_split:
                output_tokens.append(token)
                continue

            tmp_token: List[str] = []
            # tmp_token.reserve(2 * len(token))
            for char in token:
                cp = ord(char)

                if cp == 0 or cp == 0xfffd:
                    continue

                if self.is_chinese_char(cp) or self.is_alpha_or_digit(cp):
                    tmp_token.append(char)
                    continue

                cat = unicodedata.category(char)
                if self.is_control(char, cat) or self.is_mask(cp, cat):
                    continue

                if self.is_whitespace(char, cat):
                    tmp_token.append(' ')
                elif self.is_punctuation(cp, cat):
                    tmp_token.append(' ')
                    tmp_token.append(char)
                    tmp_token.append(' ')
                elif self.tokenize_emoji and self.is_emoji_char(cp):
                    tmp_token.append(' ')
                    tmp_token.append(char)
                    tmp_token.append(' ')
                else:
                    tmp_token.append(char)
            # output_tokens.extend(''.join(tmp_token).split(' '))
            for s in ''.join(tmp_token).split(' '):
                output_tokens.append(s)

        return output_tokens

    def __call__(self, text: str, clean_text: bool) -> List[str]:
        if clean_text:
            return self.tokenizer([self.clean_text(text)])[0]
        else:
            return self.tokenizer([text.split(' ')])[0]


class BertTokenizer:

    def __init__(self, tokenizer: matx.NativeObject, do_lower_case: bool, tokenize_emoji: bool, task_manager: object = None, need_clean_text: bool = True) -> None:
        '''
        默认情况下是打开的，但是线上服务的时候，可以关闭，
        '''
        self.tokenizer: matx.NativeObject = tokenizer
        self.do_lower_case: bool = do_lower_case
        self.tokenize_emoji: bool = tokenize_emoji
        self.never_split: List[str] = \
            ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
        self.clean_text_tasks: CleanText = CleanText(self.do_lower_case, self.tokenize_emoji, self.tokenizer, need_clean_text)
        self.task_manager: object = task_manager

    def __call__(self, list_text: List[str], clean_text: bool = True) -> List[List[str]]:
        spilt_texts = matx.List()
        spilt_texts.reserve(len(list_text))
        if self.task_manager is not None:
            futures = []
            for text in list_text:
                # Clean Text
                futures.append(self.task_manager.get_thread_pool().Submit(self.clean_text_tasks, text, clean_text))

            for future in futures:
                spilt_texts.append(future.get())
        else:
            for text in list_text:
                spilt_texts.append(self.clean_text_tasks(text, clean_text))

        return spilt_texts
