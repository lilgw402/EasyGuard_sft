# this is a example of creating a maxwell model
from titan.contrib.bilbo.create_bilbo_model import create_bilbo_model
import sys
import datetime

bilbo_dest = '/opt/tiger/bilbo'
sys.path.insert(0, bilbo_dest)
from utils.load_conf import load_conf



# load config


folder = '/tmp/'+datetime.datetime.now().strftime("%Y%m%d")

conf_pth='/opt/tiger/titan/titan/contrib/bilbo/example/example.conf'
config = load_conf(conf_pth, folder=folder)

# define model
model=create_bilbo_model(config)
print(model)
