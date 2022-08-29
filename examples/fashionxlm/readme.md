This is an example of language model pretrain demo.

|--mdeberta.py      # main code
|--model_utils.py   # some related utils
|--modeling_deberta_v2.py   # copy from huggingface model, fix bugs of fp16 error
|--optimizer.py 
|--lr_scheduler.py

# run on single cpu/gpu: python mdeberta.py
# run on multi gpus: 
  - local machine: /path/to/your/local/cruise/tools/TORCHRUN mdeberta.py
  - arnold: /opt/tiger/cruise/cruise/tools/TORCHRUN mdeberta.py