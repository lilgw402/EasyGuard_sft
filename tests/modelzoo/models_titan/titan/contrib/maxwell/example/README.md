Follow below commands to install maxwell dependencies
```properties
sudo apt install -y libsndfile-dev
sudo pip3 install overrides==2.8.0, addict, mmcv, sndfile==0.2.0
``` 

We import a maxwell model in this way:
```python
from titan.contrib.maxwell.create_maxwell_model import create_maxwell_model

conf_pth='titan/contrib/maxwell/example/example.json'
model=create_maxwell_model(conf_pth)
print(model)
``` 

For distributed training, users must specify the `local_rank` argument:
```python
from titan.contrib.maxwell.create_maxwell_model import create_maxwell_model

conf_pth='titan/contrib/maxwell/example/example.json'
# only rank0 will download maxwell scm
model=create_maxwell_model(conf_pth, local_rank=MY_RANK)
``` 