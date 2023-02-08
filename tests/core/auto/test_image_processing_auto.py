from easyguard.core import AutoImageProcessor, AutoModel

from transformers import AutoModelForImageClassification

extractor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

# model = AutoModelForImageClassification.from_pretrained(
#     "google/vit-base-patch16-224"
# )

model = AutoModel.from_pretrained("google/vit-base-patch16-224")
