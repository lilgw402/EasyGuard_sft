from easyguard.core import AutoImageProcessor, AutoModel


def test_hf_model():
    extractor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224"
    )

    model = AutoModel.from_pretrained("google/vit-base-patch16-224")


if __name__ == "__main__":
    test_hf_model()
