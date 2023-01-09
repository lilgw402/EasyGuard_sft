"""Try create all registered model, no pretrained, for sanity check"""
import titan
import logging


def create_all_models():
    models = titan.list_models(backend='titan')
    for model in models:
        logging.info(f"Creating {model}")
        titan.create_model(model, pretrained=False)


if __name__ == '__main__':
    create_all_models()
