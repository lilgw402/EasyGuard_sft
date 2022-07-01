import matx


class PunctuationPadding:
    """Pad a space around the punctuation."""

    def __init__(self):
        self.regex_pattern: matx.Regex = matx.Regex(
            r"([\u0021-\u002f]|[\u003a-\u0040}]|[\u005b-\u0060}]|[\u007b-\u007e]|\p{P})")
        self.replace_pattern: bytes = r" ${1} ".encode()

    def __call__(self, text: bytes) -> bytes:
        return self.regex_pattern.replace(text, self.replace_pattern)
