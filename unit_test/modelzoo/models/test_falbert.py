import unittest

# test module

TEST_FLAGS = ["all"]


class TestFashionSwin(unittest.TestCase):
    @unittest.skipUnless(
        "all" in TEST_FLAGS or "falbert" in TEST_FLAGS, "just do it"
    )
    def test_falbert(self):
        from easyguard import AutoModel, AutoProcessor

        text_input = None
        image_input = None
        archive = "falbert-no-weights"
        processor = AutoProcessor.from_pretrained(archive)
        # inputs = processor(dummy_input)
        model = AutoModel.from_pretrained(archive)
        model.eval()
        # ouputs = model(inputs)
        # print(ouputs)

    @unittest.skipUnless(
        "all" in TEST_FLAGS or "falbert_local" in TEST_FLAGS,
        "just do it",
    )
    def test_falbert_local(self):
        from easyguard import AutoModel, AutoProcessor

        text_input = None
        image_input = None
        archive = "/root/.cache/easyguard/models/falbert/c05c83c4d58c86e36aa37a35a1fbc966edf6dd8948516631804af54f0bab211c"
        processor = AutoProcessor.from_pretrained(archive)
        # inputs = processor(dummy_input)
        model = AutoModel.from_pretrained(archive)
        model.eval()
        # ouputs = model(inputs)
        # print(ouputs)


if __name__ == "__main__":
    unittest.main()
