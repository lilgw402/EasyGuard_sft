# this is a example of creating a maxwell model
from titan.contrib.maxwell.create_maxwell_model import create_maxwell_model

conf_pth='/opt/tiger/titan/titan/contrib/maxwell/example/example_bigbird.json'
model=create_maxwell_model(conf_pth)
print(model)
