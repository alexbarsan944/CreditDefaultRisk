"""
Utilities for the Credit Default Risk web application.

This package contains utility functions used by the web application.
"""
from credit_risk.web.utils.config import get_config
from credit_risk.web.utils.page_config import set_page_config
from credit_risk.web.utils.gpu import is_gpu_available
from credit_risk.web.utils.sidebar import setup_sidebar
from credit_risk.web.utils.sampling import (
    apply_sampling_profile,
    create_train_test_split,
    create_stratified_folds
)

__all__ = [
    "get_config",
    "set_page_config",
    "is_gpu_available",
    "setup_sidebar",
    "apply_sampling_profile",
    "create_train_test_split",
    "create_stratified_folds"
]
