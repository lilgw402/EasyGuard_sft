
import sys
from pathlib import Path
from titan.utils.misc import download_and_extract_scm
from torch.distributed import is_initialized, barrier






def init_model(config,local_rank=0):
    from model.model_factory import ModelFactory
    from utils.feature_provider import FeatureProviderFactory
    from utils.device import init_device, put_data_into_device

    device, n_gpu = init_device()

    parameters = eval(config.get('model_instance', 'parameters'))
    model_type = config.get('model_instance', 'model_type')
    model = ModelFactory.get(model_type, None)(parameters,device)

    # Training process don't need extra information, but testing process should use it.
    # train_feature_provider = FeatureProviderFactory.get(model_type, None)(extra=False)
    # test_feature_provider = FeatureProviderFactory.get(model_type, None)(extra=True)

    return model#, train_feature_provider, test_feature_provider

def download_model(config):
    from utils.file_sync_util import  pull_files, pull_file_from_hdfs_path
    input_model_dir = config.get('workflow', 'input_model_dir') if config.has_option('workflow', 'input_model_dir') else None

    if input_model_dir:
        input_model_dir = pull_files(input_model_dir)



def create_bilbo_model(config, local_rank=0):
    exist_bilbo = any(path.endswith('bilbo') for path in sys.path)
    bilbo_dest = '/opt/tiger/bilbo'
    if not exist_bilbo:
        if not Path(bilbo_dest).exists() and local_rank == 0:
            # Download SCM
            ## TODO Junda please fix SCM
            # download_and_extract_scm(bilbo_dest, 'data.content_security.maxwell', '1.0.0.1038')
            pass
        if is_initialized():
            barrier()
    
    sys.path.insert(0, bilbo_dest)

    # download_model(config)
    # model , train_feature_provider, test_feature_provider=init_model(config, local_rank=local_rank)
    model=init_model(config, local_rank=local_rank)

    return model#, train_feature_provider, test_feature_provider
