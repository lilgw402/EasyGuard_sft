from typing import Dict
from typing import Any

from fex import _logger as logger

try:
    from matx.pypi import farmhash
except Exception as e:
    logger.warning(f'matx is not installed: {e}. you can install it from: https://bytedance.feishu.cn/wiki/wikcnS7vYo5ZwvCImghxdLUYlag')


class Vocab:
    def __init__(self, vocab: str, unk: str, offset: int = 0, num_oov_buckets: int = 0) -> None:
        self.v: Dict[str, int] = {}
        self.v_list = []
        self.unk_id: int = 0
        self.offset: int = offset
        self.num_oov_buckets: int = num_oov_buckets
        i = 0
        fd = open(vocab)
        for line in fd:
            term = line.strip()
            self.v[term] = i
            self.v_list.append(term)
            if term == unk:
                self.unk_id = i
            i += 1
        fd.close()
        assert unk in self.v, "vocab should contain unk"
        self.v_len: int = len(self.v)

    def lookup(self, token: str) -> int:
        lookup_id = self.v.get(token, self.unk_id)
        if self.num_oov_buckets > 0 and lookup_id == self.unk_id:
            return farmhash.fingerprint64_mod(token, self.num_oov_buckets) + self.v_len + self.offset
        else:
            return lookup_id + self.offset

    def id2token(self, id: int) -> str:
        return self.v_list[id]

    def len(self, with_oov: bool = False) -> int:
        if not with_oov:
            return self.v_len
        else:
            return self.v_len + self.num_oov_buckets

    def __call__(self, token: str) -> int:
        return self.lookup(token)
