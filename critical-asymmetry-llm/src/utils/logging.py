import logging
import sys


def get_logger(name="caa", level="INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger
