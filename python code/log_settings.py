import logging

logger = logging.getLogger("Demo")
c_handler = logging.StreamHandler()

logger.setLevel("INFO")
c_format = logging.Formatter("%(asctime)s %(levelname)s %(module)s  %(message)s")
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)