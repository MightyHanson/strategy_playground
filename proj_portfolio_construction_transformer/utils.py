# utils.py

import logging
import os

def setup_logging(output_dir, log_file='app.log'):
    """
    Set up logging configuration.

    Args:
        output_dir (str): Directory where the log file will be saved.
        log_file (str, optional): Name of the log file. Defaults to 'app.log'.
    """
    log_path = os.path.join(output_dir, log_file)
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logging.info("Logging is set up.")
