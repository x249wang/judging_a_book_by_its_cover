import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

c_handler = logging.StreamHandler()
c_handler.setFormatter(formatter)

f_handler = logging.FileHandler(filename="ml/logging.log", mode="a")
f_handler.setFormatter(formatter)

logger.addHandler(c_handler)
logger.addHandler(f_handler)
