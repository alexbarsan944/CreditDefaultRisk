import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from typing import List, Optional, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler

from credit_risk.web.ui_components import (
    display_info_box, 
    parameter_help, 
    educational_tip,
    display_dataframe_with_metrics,
    plot_missing_values,
    section_header,
)
from credit_risk.utils.streamlit_utils import prepare_dataframe_for_streamlit
from credit_risk.utils.logging_utils import get_logger
from credit_risk.web.exploratory_analysis_tab import SuppressWarnings, render_exploratory_analysis_tab

# Configure logging
logger = get_logger(__name__)

# Create a context manager to temporarily suppress warnings
class SuppressWarnings:
    def __enter__(self):
        self.original_filters = warnings.filters.copy()
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in')
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero')
        return self
        
    def __exit__(self, *args):
        warnings.filters = self.original_filters

# ... existing code ... 