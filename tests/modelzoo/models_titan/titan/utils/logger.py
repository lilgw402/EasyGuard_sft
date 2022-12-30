import logging
import sys

logger = logging.getLogger("bytedtitan")
if len(logger.handlers) == 0:
    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setFormatter(
        logging.Formatter("[%(asctime)s] - [%(levelname)s] - %(message)s")
    )
    logger.addHandler(log_handler)
    log_handler.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
