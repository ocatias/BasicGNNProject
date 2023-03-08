"""
Tests run_experiment.py by performing hyper parameter tuning on CSL and ogbg-molesol
We use these two datasets as they are both small and one uses cross-validation and the other a train, validation and test split
"""

import os
import unittest
import shutil
import glob

from Misc.config import config
from Exp.run_experiment import main as run_experiment

def check_if_error_files_exist(path):
    """
    Ensure the experiment has not thrown any error
    """
    error_files = glob.glob(os.path.join(path, "**", "error_*.json"))
    print(error_files)
    assert len(error_files) == 0
    

class RunExpTest(unittest.TestCase):
    def test_run_exp_train_val_test(self):
        """
        run_experiment on a dataset with a train, val, test split
        """
        print(os.path.join(config.TESTS_PATH, "run_exp_test_config.yaml"))
        run_experiment(passed_args = {
            "-grid": os.path.join(config.TESTS_PATH, "run_exp_test_config.yaml"),
            "-dataset": "ogbg-molesol",
            "--candidates": "2",
            "--repeats": "2"
        })
        check_if_error_files_exist(os.path.join(config.RESULTS_PATH, "ogbg-molesol_run_exp_test_config.yaml"))
        
    def test_run_exp_cross_val(self):
        """
        run_experiment on a dataset with cross-validation
        """
        print(os.path.join(config.TESTS_PATH, "run_exp_test_config.yaml"))
        run_experiment(passed_args = {
            "-grid": os.path.join(config.TESTS_PATH, "run_exp_test_config.yaml"),
            "-dataset": "CSL",
            "--candidates": "2",
            "--repeats": "2",
            "--folds": "5"
        })
        check_if_error_files_exist(os.path.join(config.RESULTS_PATH, "CSL_run_exp_test_config.yaml"))

if __name__ == '__main__':
    # Remove processed results folder from previously running this test
    
    for dir_name in ["ogbg-molesol_run_exp_test_config.yaml", "CSL_run_exp_test_config.yaml"]:
        dir_path = os.path.join(config.RESULTS_PATH, dir_name)
        
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)

    # Run tests
    unittest.main()