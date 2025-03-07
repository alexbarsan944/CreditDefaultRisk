# Unit Tests for Credit Default Risk Application

This directory contains unit tests for the Credit Default Risk application. These tests are designed to verify the correctness of various components and help identify and debug issues.

## Running Tests

You can run the tests using the provided `run_tests.py` script at the root of the project:

```bash
# Run all tests
./run_tests.py

# Run tests with verbose output
./run_tests.py -v

# Run specific test file
./run_tests.py --path tests/test_features/test_model_training.py
```

Or you can use pytest directly:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_features/test_model_training.py

# Run tests with verbose output
pytest -v

# Run a specific test
pytest tests/test_features/test_model_training.py::TestModelTraining::test_check_data_quality
```

## Test Directories

- `test_features/` - Tests for feature engineering and model training functionality
- `test_data/` - Test data fixtures for testing

## Adding New Tests

When adding new tests, follow these guidelines:

1. Place tests in the appropriate subdirectory based on the functionality being tested
2. Use the `unittest` framework and extend `unittest.TestCase`
3. Name test files with the prefix `test_` 
4. Name test methods with the prefix `test_`
5. Include docstrings explaining what each test is checking

## Debugging NaN Issues

The tests include specific functionality for detecting and handling NaN values in data:

- `test_hyperopt_nan_issue.py` - Tests for detecting and handling NaN values in hyperparameter optimization
- `test_gpu_and_nan_handling.py` - Tests for GPU error handling and NaN detection

## Test Fixtures

Test fixtures provide standardized test data. Main fixtures include:

- Clean datasets with no issues
- Datasets with NaN values
- Datasets with infinity values
- Datasets with outliers

## Mocking External Dependencies

To speed up tests and avoid external dependencies, many tests use mocking to simulate external systems:

- `unittest.mock.patch` for patching functions and classes
- `unittest.mock.MagicMock` for creating mock objects

## Coverage Reporting

To generate a coverage report, run:

```bash
pytest --cov=src tests/
```

Or with a HTML report:

```bash
pytest --cov=src --cov-report=html tests/
``` 