from random import sample
import titan
from titan import TOSHelper

tos_helper = TOSHelper(bucket='titan-modelzoo-public', access_key='BW7H90QZ6H7YR0U92WWM')
net = titan.create_model('resnet50', pretrained=True, tos_helper=tos_helper)

def create_pretrained_sample_models():
    sample_models = ['resnet50', 'convnext_tiny', 'swin_tiny', 'vit_tiny', ]
    for model in sample_models:
        m = titan.create_model(model, pretrained=True, tos_helper=tos_helper)



if __name__ == '__main__':
    create_pretrained_sample_models()
