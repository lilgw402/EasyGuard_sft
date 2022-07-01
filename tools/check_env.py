# 检查环境
from fex import _logger as logger
import importlib

TEXTOP_ENV = False

env = {
    x: False
    for x in
    ['tvm', 'xpert', 'torch', 'torch_cuda'
     'fex', 'textops', 'visionops']
}

try:
    import xperf.ops
    xperf.ops.load_ft_torch()
    logger.info(f'{"xpert exist":40s} path [{xperf.ops.__file__}]')
    env['xperf'] = True
except Exception as e:
    logger.error(e)
    logger.warning('xpert not exist')

try:
    import tvm
    from tvm.contrib.pt_op import PyTorchTVMModule
    logger.info(f'{"tvm exist":40s} path [{tvm.__file__}]')
    env['tvm'] = True
except Exception as e:
    logger.error(e)
    logger.warning('tvm not exist')

try:
    import torch
    logger.info(
        f'{"torch exist":20s} {f"version {torch.__version__}":20s} path [{torch.__file__}]'
    )
    env['torch'] = True

    if torch.cuda.is_available():
        logger.info('torch cuda is available')
    else:
        logger.warning('torch cuda is not available')
    env['torch_cuda'] = True

    logger.info('cuda version: ' + torch.version.cuda)
except Exception as e:
    logger.error(e)
    logger.warning('torch not exist')

for name in ['textops', 'visionops', 'rich', 'fex']:
    try:
        _tmp = importlib.import_module(name)
        logger.info(
            f'{name+" exist":20s} {"version " + getattr(_tmp, "__version__", "none"):20s} path [{_tmp.__file__}]'
        )
        env[name] = {
            'name': _tmp.__name__,
            'version': getattr(_tmp, '__version__', 'none'),
            'path': _tmp.__file__
        }
    except Exception as e:
        logger.warning(f'{name} not exist')
