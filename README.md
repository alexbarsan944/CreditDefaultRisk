# Credit Default Risk Prediction

A modern Python library for credit default risk prediction using automated feature engineering with GPU acceleration and MLflow tracking.

## Overview

This package provides a comprehensive pipeline for credit risk prediction with a focus on automated feature engineering using [featuretools](https://docs.featuretools.alteryx.com/). The library includes:

- Efficient data loading and preprocessing
- Memory optimization for handling large datasets
- GPU-accelerated data processing (with fallback to CPU)
- Automated feature engineering with featuretools
- Manual feature creation based on domain knowledge
- Feature storage and versioning
- MLflow experiment tracking
- Interactive web interface
- Development mode for quick iteration
- Comprehensive testing and documentation

## Installation

### From PyPI

```bash
pip install credit-risk
```

### From Source

```bash
git clone https://github.com/yourusername/credit-risk.git
cd credit-risk
pip install -e .
```

For development, install with additional dependencies:

```bash
pip install -e ".[dev]"
```

For GPU acceleration, install with:

```bash
pip install -e ".[gpu]"
```

For the web interface, install with:

```bash
pip install -e ".[web]"
```

For all features, install with:

```bash
pip install -e ".[dev,gpu,web]"
```

### RAPIDS Installation (for full GPU acceleration)

For full GPU acceleration with NVIDIA GPUs, install the RAPIDS suite:

```bash
# Using conda (recommended)
conda install -c rapidsai -c conda-forge cudf=23.04 python=3.10 cudatoolkit=11.8
```

## Quick Start

### Running the Feature Engineering Pipeline with GPU Acceleration

```python
from credit_risk import (
    PipelineConfig, 
    RunMode,
    ProcessingBackend,
    GPUDataLoader, 
    DataPreprocessor, 
    FeatureEngineer, 
    FeatureStore,
    ExperimentTracker
)

# Configure the pipeline with GPU acceleration
config = PipelineConfig(
    run_mode=RunMode.DEVELOPMENT,  # Use a smaller data sample
    gpu=PipelineConfig().gpu
)
config.gpu.processing_backend = ProcessingBackend.AUTO  # Auto-detect best available backend

# Initialize MLflow tracking
tracker = ExperimentTracker(config=config)

# Create a GPU-accelerated data loader
loader = GPUDataLoader(config=config)
print(f"Using {loader.get_backend_name()} backend for data processing")

# Load and preprocess data
datasets = loader.load_all_datasets()
preprocessor = DataPreprocessor()
app_df = datasets["application_train"]
app_df = preprocessor.drop_low_importance_columns(app_df)
app_df = preprocessor.handle_missing_values(app_df)
datasets["application_train"] = app_df

# Start MLflow tracking run
with tracker.start_run(run_name="feature_engineering"):
    # Log parameters
    tracker.log_params({
        "run_mode": config.run_mode.value,
        "backend": loader.get_backend_name(),
        "max_depth": config.features.max_depth
    })
    
    # Create features
    engineer = FeatureEngineer(config=config)
    
    # Add manual features based on domain knowledge
    app_df = engineer.create_manual_features(
        app_df=app_df,
        bureau_df=datasets.get("bureau"),
        previous_df=datasets.get("previous_application"),
        installments_df=datasets.get("installments_payments")
    )
    
    # Generate automated features
    features_df = engineer.generate_features(
        datasets=datasets,
        max_depth=config.features.max_depth,
        verbose=True
    )
    
    # Log metrics
    tracker.log_metrics({
        "num_features": features_df.shape[1],
        "num_samples": features_df.shape[0]
    })
    
    # Save features to feature store
    store = FeatureStore(config=config)
    store.save_feature_set(
        features=features_df,
        name="credit_risk_features",
        description="Credit risk prediction features",
        tags=["credit_risk", "featuretools", loader.get_backend_name()]
    )
```

### Command Line Interface

You can also run the GPU-accelerated feature engineering pipeline from the command line:

```bash
# Run in development mode with automatic backend selection
credit-risk-features --dev

# Run with specific backend and parameters
credit-risk-features --backend gpu --max-depth 3 --output my_features

# Run the advanced GPU example
python -m src.examples.feature_engineering_with_gpu --backend gpu --max-depth 2 --run-mode development
```

### Web Interface

Launch the interactive web interface for exploring data and running experiments:

```bash
# Start the Streamlit web app
credit-risk-app

# Or run directly with Python
python -m src.credit_risk.web.app
```

## GPU Acceleration

The library supports multiple GPU acceleration backends:

1. **RAPIDS cuDF** - Full GPU acceleration using NVIDIA GPUs
2. **Polars GPU** - GPU-accelerated operations via CUDA
3. **Pandas (CPU fallback)** - Automatic fallback when GPU is not available

The processing backend automatically selects the best available option:

```python
from credit_risk import ProcessingBackend, GPUDataLoader

# Auto-detect best available backend
loader = GPUDataLoader(backend=ProcessingBackend.AUTO)

# Force GPU backend (will raise error if not available)
loader = GPUDataLoader(backend=ProcessingBackend.GPU)

# Force CPU backend
loader = GPUDataLoader(backend=ProcessingBackend.CPU)

# Check if GPU is being used
is_using_gpu = loader.is_gpu_available()
backend_name = loader.get_backend_name()
print(f"Using {backend_name} backend (GPU: {is_using_gpu})")
```

## MLflow Experiment Tracking

The package integrates with MLflow for experiment tracking:

```python
from credit_risk import ExperimentTracker, track_with_mlflow

# Initialize the tracker
tracker = ExperimentTracker()

# Track a run manually
with tracker.start_run(run_name="my_experiment"):
    # Log parameters
    tracker.log_params({"param1": "value1", "param2": 42})
    
    # Log metrics
    tracker.log_metrics({"accuracy": 0.95, "f1": 0.92})
    
    # Log artifacts
    tracker.log_artifact("path/to/file.txt")

# Or use the decorator
@track_with_mlflow(run_name="my_function")
def my_function(arg1, arg2):
    # Function code here
    return result
```

## Data Requirements

The pipeline expects the following data files in the configured data directory:

- `application_train.csv`: Main application data with target variable
- `application_test.csv`: Test application data
- `bureau.csv`: Credit bureau data
- `bureau_balance.csv`: Monthly credit bureau data
- `credit_card_balance.csv`: Credit card balance data
- `installments_payments.csv`: Installment payment data
- `POS_CASH_balance.csv`: POS and cash loan balance data
- `previous_application.csv`: Previous application data

## Project Structure

```
credit_risk_prediction/
├── src/
│   ├── credit_risk/              # Main package
│   │   ├── __init__.py           # Package exports
│   │   ├── config.py             # Configuration settings
│   │   ├── data/                 # Data handling
│   │   │   ├── loader.py         # Data loading
│   │   │   ├── gpu_loader.py     # GPU-accelerated loader
│   │   │   ├── preprocessor.py   # Data preprocessing
│   │   │   └── optimizers.py     # Memory optimization
│   │   ├── features/             # Feature engineering
│   │   │   ├── engineer.py       # Feature creation
│   │   │   ├── entity_builder.py # Entity set builder
│   │   │   └── feature_store.py  # Feature storage
│   │   ├── utils/                # Utilities
│   │   │   ├── processing_strategy.py  # GPU/CPU strategies
│   │   │   ├── mlflow_tracking.py      # MLflow integration
│   │   │   └── logging_config.py       # Logging configuration
│   │   └── web/                  # Web interface
│   │       ├── __init__.py       # Web module exports
│   │       └── app.py            # Streamlit web application
│   └── examples/                 # Example scripts
│       └── feature_engineering_with_gpu.py  # GPU example
├── tests/                        # Unit tests
│   ├── test_data/                # Data tests
│   └── test_features/            # Feature tests
├── data/                         # Data directory
│   ├── raw/                      # Raw data files
│   ├── processed/                # Processed data
│   └── features/                 # Extracted features
├── pyproject.toml                # Package configuration
└── README.md                     # You are here
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=credit_risk

# Run specific test file
pytest tests/test_data/test_optimizers.py
```

### Code Formatting

```bash
# Format code with black
black src/ tests/

# Sort imports
isort src/ tests/

# Check types
mypy src/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 

