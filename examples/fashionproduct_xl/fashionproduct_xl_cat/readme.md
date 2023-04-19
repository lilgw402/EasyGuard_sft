# This is an example of fashionproduct-xl finetune demo.

```bash
# 0. pip3 install ptx if needed
pip3 install https://d.scm.byted.org/api/v2/download/nlp.lib.ptx2_1.0.0.176.tar.gz
# 1. can dump default configs as initial config file (in your local machine)
python3 example/fashoinproduct_xl/run_pretrain.py --print_config > example/fashoinproduct/default_config.yaml
# 2. modify the file
cat example/fashoinproduct_xl/default_config.yaml
# 3. load the modified config back
python3 example/fashoinproduct_xl/run_pretrain.py --config example/fashoinproduct/default_config.yaml
or
python3 example/fashoinproduct_xl/run_pretrain.py --config hdfs://path/to/your/default_config.yaml 
# 4. customize extra configs manually
python3 example/fashoinproduct_xl/run_pretrain.py --config example/fashoinproduct/default_config.yaml --model.hidden_size=1024
```
