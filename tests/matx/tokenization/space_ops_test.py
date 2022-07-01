
import matx
import unittest
from fex.matx.tokenization.matx_caster.space_ops import PunctuationPadding


class TestSpace(unittest.TestCase):
    @unittest.skip("Data Path not exist")
    def test_PunctuationPadding(self):
        op = matx.script(PunctuationPadding)()
        for text, res in [("hello gg?world!aa+ss？是？！dddd/ddddd:444@", "hello gg ? world ! aa + ss ？ 是 ？  ！ dddd / ddddd : 444 @ "),
                          ("aaa[sssgg],^rrr`111{{}}bla~~", "aaa [ sssgg ]  ,  ^ rrr ` 111 {  {  }  } bla ~  ~ ")]:
            r = op(text.encode("utf-8"))
            self.assertEqual(r.decode("utf-8"), res)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
