import os
import time
from datetime import datetime


def create_log_dir(log_dir=None):
    start = time.time()
    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d-%H-%M-%S')

    log_dir = log_dir if log_dir is not None else "log"
    full_log_dir = os.path.join(log_dir, "log" + now_str)
    if not os.path.exists(full_log_dir):
        os.makedirs(full_log_dir)

    return full_log_dir
