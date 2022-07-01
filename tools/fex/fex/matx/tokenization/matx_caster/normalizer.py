import matx
from typing import List
import unicodedata


class TextCleaner:
    """TextCleaner impl by matx."""

    def __init__(self) -> None:
        self.white_regex: matx.Regex = matx.Regex(r"[ \t\n\r\p{Zs}]")
        self.control_regex: matx.Regex = matx.Regex(
            r"[\u0000\ufffd\p{Cc}\p{Cf}\p{Mn}]")

        self.space: bytes = " ".encode()
        self.empty: bytes = "".encode()

    def __call__(self, text: bytes) -> bytes:
        t = self.white_regex.replace(text, self.space)
        return self.control_regex.replace(t, self.empty)


class CaseNormalizer:
    def __init__(self, do_lowercase: bool = False, unicode_norm: str = '') -> None:
        self.do_lowercase: bool = do_lowercase

    def __call__(self, text: bytes) -> bytes:
        if self.do_lowercase:
            return text.lower()
        else:
            return text
