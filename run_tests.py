#!/usr/bin/env python
"""
Run unit tests for the Credit Default Risk project.
"""

import pytest
import sys
import os
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_tests(test_path=None, verbose=False, debug=False):
    """
    Run the unit tests.
    
    Parameters:
    -----------
    test_path : str, optional
        Specific test path to run, by default None (run all tests)
    verbose : bool, optional
        Whether to show verbose output, by default False
    debug : bool, optional
        Whether to run in debug mode, by default False
    """
    # Set up arguments for pytest
    args = []
    
    # Add verbosity
    if verbose:
        args.append("-v")
    
    # Add test path if specified
    if test_path:
        args.append(test_path)
    else:
        args.append("tests/")
    
    # Add debugging options
    if debug:
        args.append("--pdb")
    
    logger.info(f"Running tests with arguments: {args}")
    
    # Run pytest
    return pytest.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Credit Default Risk unit tests")
    parser.add_argument("--path", type=str, help="Specific test path to run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("-d", "--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Run the tests
    exit_code = run_tests(test_path=args.path, verbose=args.verbose, debug=args.debug)
    
    # Exit with the pytest exit code
    sys.exit(exit_code) 