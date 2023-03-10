import unittest

from Exp.run_model import run as run_model
from Exp.parser import parse_args
from Exp.preparation import load_dataset
from Misc.config import config
from Scripts.clean_datasets_dir import main as clean_datasets_dir

models = ["GIN", "GCN", "MLP"]
datasets = ["CSL", "ZINC", "ogbg-molhiv", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molclintox", "ogbg-molbbbp", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-mollipo"]
smaller_number_of_datasets = ["CSL", "ZINC", "ogbg-molhiv"]
epochs = 2
use_tracking = False

class ModelTrainingTest(unittest.TestCase):
    def test_model_training(self):
        """
        Trains every model on every dataset for 2 epochs.
        """
        for model in models:
            for dataset in datasets:
                with self.subTest(msg=f"Testing {model} on {dataset}"):
                    run_model(passed_args = {
                        "--model": model,
                        "--dataset": dataset,
                        "--epochs": str(epochs),
                        "--tracking": "1" if use_tracking else "0"
                    })
                    
class DropFeatureTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(DropFeatureTest, self).__init__(*args, **kwargs)
        # Remove processed datasets to allows us to test generating them
        clean_datasets_dir()  
        
    def test_dropping_features(self):
        """
        Drops features from all datasets
        """
        for dataset in datasets:
            with self.subTest(msg=f"Testing dropping features on {dataset}"):
                args = { 
                    "--dataset": dataset,
                    "--drop_feat": 1                    
                }
                load_dataset(parse_args(args), config)
                
    def test_training_on_dropped_feats(self):
        """
        Trains every model on 3 datasets with dropped features 
        """
        for model in models:
            for dataset in smaller_number_of_datasets:
                with self.subTest(msg=f"Testing {model} on {dataset} with dropped features"):
                    args =  {
                        "--model": model,
                        "--dataset": dataset,
                        "--epochs": str(epochs),
                        "--tracking": "1" if use_tracking else "0",
                        "--drop_feat": 1  
                    }
                    run_model(passed_args = args)
    
if __name__ == '__main__':
    # Run tests
    unittest.main()
