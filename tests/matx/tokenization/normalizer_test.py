import matx
import unittest
from fex.matx.tokenization.matx_caster.normalizer import TextCleaner


class TestNormalize(unittest.TestCase):
    @unittest.skip("Data Path not exist")
    def test_text_clean(self):
        cleaner = matx.script(TextCleaner)()
        
        text = "\u0000\ufffd你好\u0020\t\u0600中国".encode("utf-8")
        r = cleaner(text)
        self.assertEqual(r.decode("utf-8"), "你好  中国")

        text = "he\u0015llo\t gg\n\twor⃦ld! aa".encode("utf-8")
        r = cleaner(text)
        self.assertEqual(r.decode("utf-8"), "hello  gg  world! aa")

        text = "hello\t \rgg\n\twor\ufffdld!\x00a\u200da".encode("utf-8")
        r = cleaner(text)
        self.assertEqual(r.decode("utf-8"), "hello   gg  world!aa")

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
