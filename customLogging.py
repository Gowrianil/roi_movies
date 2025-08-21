#!/usr/bin/env python3

## author Rahul G
import logging
from datetime import datetime
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG) 
formatter = logging.Formatter("[%(filename)s:%(lineno)d][%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S.%f")

def custom_time(*args):
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

formatter.formatTime = custom_time
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
