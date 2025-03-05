import logging

logger = logging.getLogger("credit_risk_utils")

def is_gpu_available():
    """
    Check if GPU is available for model training.
    
    Returns
    -------
    bool
        True if GPU is available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        logger.warning("PyTorch not installed, cannot check GPU availability")
        return False 