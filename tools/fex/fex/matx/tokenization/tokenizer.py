from typing import List

import string
import random
import os
from fex import _logger as logger
from fex.utils.hdfs_io import hcopy

from .matx_caster import TextCleaner, CaseNormalizer, PunctuationPadding, WordpieceTokenizer, Vocab

try:
    from text_cutter import Cutter
except Exception as e:
    logger.warning(f'text_cutter import failed: {e}. install by: `pip3 install https://d.scm.byted.org/api/v2/download/search.nlp.libcut_py_matx4_2.3.0.8.tar.gz`')

try:
    from libcut_data_zh_20200827 import data_path as libcut_data_path
except Exception as e:
    logger.warning(
        f'libcut_data_zh_20200827 import failed: {e}. install by: `pip install http://d.scm.byted.org/api/v2/download/search.nlp.libcut_data_zh_20200827_1.0.0.2.tar.gz`')
    libcut_data_path = None


class MatxBertTokenizer:
    def __init__(self,
                 vocab_path: str,
                 do_cut: bool = False,
                 lower_case: bool = False,
                 max_tokens_per_input: int = 256,
                 unk_token: str = '[UNK]',
                 wordpiece_type: str = 'bert',
                 unk_kept: bool = False) -> None:
        """
        matx 版本的 BertTokenzier。
        vocab_path: tokenizer 用的词表路径。如果是hdfs路径，回拉到 /tmp 下
        do_cut: 是否做切词操作
        lower_case: 是否全转小写
        max_tokens_per_input: 最长多少个token
        wordpiece_type: wordpiece 的类型，一般用 `bert` 就行，如果需要对齐caster，可以用`caster-bert`
        unk_token: 如果是未知的词，用什么token 替换
        unk_kept: 是否保持不认识的那个term。如果 unk_kept=true, 遇到不认识的term会保留，如果 unk_kept=false，遇到不认识的term会换成固定的 unk_token。比如 `毀 灭` 当unk_kept=true，依然是 `毀 灭`，当unk_kept=false，则是 `[UNK] 灭`。一般在有unk hash 的策略时有用

        tokenizer = MatxBertTokenizer(vocab_path=vocab_path,
                                      do_cut=True,
                                      lower_case=True,
                                      unk_token='[UNK]',
                                      wordpiece_type='bert')

        """
        self.cleaner: TextCleaner = TextCleaner()
        self.normalizer: CaseNormalizer = CaseNormalizer(True)
        self.punc_padding: PunctuationPadding = PunctuationPadding()
        self.do_cut = do_cut
        if libcut_data_path and self.do_cut:
            self.cutter: Cutter = Cutter("CRF_LARGE", libcut_data_path)
        else:
            logger.warning('MatxBertTokenizer will not cut text, please make sure you already do text cut')
        vocab_path = self.fetch_vocab_path_from_hdfs(vocab_path)
        self.world_piece: WordpieceTokenizer = WordpieceTokenizer(
            vocab_path,
            max_tokens_per_input=max_tokens_per_input,
            lower_case=lower_case,
            unk_token=unk_token,
            wordpiece_type=wordpiece_type,
            unk_kept=unk_kept)
        self.vocab = Vocab(vocab_path, unk=unk_token)

    def fetch_vocab_path_from_hdfs(self, vocab_path):
        """
        只支持load 本地的词表，因此这里需要判断一下词表是不是hdfs路径，如果是的话拷贝到 /tmp 下
        """
        def random_string(length):
            salt = ''.join(random.sample(string.ascii_letters + string.digits, length))
            return salt

        if vocab_path.startswith('hdfs'):
            rank = str(os.environ.get('RANK') or 0)
            local_vocab_path = os.path.join('/tmp', f"{vocab_path.split('/')[-1]}_{random_string(10)}_{rank}")  # 按 vocab_file_name, random_string, rank 作为名字
            logger.info(f'{vocab_path} is on hdfs, copy it to {local_vocab_path}')
            hcopy(vocab_path, local_vocab_path)
            vocab_path = local_vocab_path

        return vocab_path

    def __call__(self, text: bytes) -> List[str]:
        """
        输入需要是bytes
        所以如果输入是str，这里 encode一下。
        但在cutter和后面的tokenzier的输入是str，所以需要decode回去
        如：
        ['关于', '##美', '##食', '##单', '##词', '##之', '##二', '#', '亲子', '##英', '##语', '#', '每日', '##英', '##语', '#', '英语', '##单', '##词']
        """
        if isinstance(text, str):
            text = text.encode()
        t = self.cleaner(text)
        t = self.normalizer(t)
        t = self.punc_padding(t)
        if isinstance(t, bytes):
            t = t.decode()
        if self.do_cut:
            terms = self.cutter.cut(t, "FINE")
        else:
            terms = t.split()
        return self.world_piece(terms)

    def tokens2idx(self, tokens: List[str]) -> List[int]:
        return [self.vocab.lookup(t) for t in tokens]

    def idx2tokens(self, ids: List[int]) -> List[str]:
        return [self.vocab.id2token(i) for i in ids]
