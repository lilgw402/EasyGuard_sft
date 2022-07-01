import os
import logging


logger = logging.getLogger(__name__)

USE_PTX_TRANSFORMER = bool(int(os.getenv('FEX_USE_PTX_TSFM', '0')))

USE_PTX_TRANSFORMER_CONF_ALBERT = os.getenv('FEX_USE_PTX_TSFM_CONF_ALBERT', '')
USE_PTX_TRANSFORMER_CONF_ALBERT = [i.strip() for i in USE_PTX_TRANSFORMER_CONF_ALBERT.split(',')]
USE_PTX_TRANSFORMER_CONF_ALBERT = [i for i in USE_PTX_TRANSFORMER_CONF_ALBERT if i]
USE_PTX_TRANSFORMER_CONF_ALBERT = {i.split(':')[0].strip(): bool(int(i.split(':')[1].strip())) for i in USE_PTX_TRANSFORMER_CONF_ALBERT}

USE_PTX_TRANSFORMER_CONF_VIT = os.getenv('FEX_USE_PTX_TSFM_CONF_VIT', '')
USE_PTX_TRANSFORMER_CONF_VIT = [i.strip() for i in USE_PTX_TRANSFORMER_CONF_VIT.split(',')]
USE_PTX_TRANSFORMER_CONF_VIT = [i for i in USE_PTX_TRANSFORMER_CONF_VIT if i]
USE_PTX_TRANSFORMER_CONF_VIT = {i.split(':')[0].strip(): bool(int(i.split(':')[1].strip())) for i in USE_PTX_TRANSFORMER_CONF_VIT}

try:
    from ptx.ops.transformer import TransformerEncoderLayer
except ImportError:
    TransformerEncoderLayer = None
    if USE_PTX_TRANSFORMER:
        logger.warning('Failed to import ptx ops')
        USE_PTX_TRANSFORMER = False
