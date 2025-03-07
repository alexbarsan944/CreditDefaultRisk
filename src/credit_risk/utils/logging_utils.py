import logging
import sys
from pathlib import Path
from typing import Optional

def get_logger(name: str, level: int = logging.INFO, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Parameters
    ----------
    name : str
        Name of the logger
    level : int
        Logging level (default: INFO)
    log_file : Optional[Path]
        Path to log file, if file logging is desired
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if logger already has handlers to avoid duplicate handlers
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler if log_file is provided
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger 