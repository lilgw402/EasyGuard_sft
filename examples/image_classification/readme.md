# This is an example of fashion-swin finetune demo.

```python
# modify the config file
vi examples/image_classification/default_config.yaml
# if no config file found, can dump default configs as initial file (in your local machine)
python3 examples/image_classification/run_model_finetuning.py  --print_config > examples/image_classification/default_config.yaml
```

**feature extraction demo:**
- `python3 examples/image_classification/feature_extractor.py`

**train on single cpu/gpu:**
- `python3 examples/image_classification/run_model_finetuning.py --config examples/image_classification/default_config.yaml`

**train on multi gpus:**
- local machine: `/path/to/your/local/EasyGuard/tools/TORCHRUN examples/image_classification/run_model_finetuning.py --config examples/image_classification/default_config.yaml`
- arnold: `/opt/tiger/EasyGuard/tools/TORCHRUN examples/image_classification/run_model_finetuning.py --config examples/image_classification/default_config.yaml`