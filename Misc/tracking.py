"""
Supports different ways to track experiments:
    - Locally in text files
    - Weights and Biases
You can set the tracking you want to use in Configs/config.yaml
"""

import datetime
import os
import json

from Misc.config import config

class Tracker:
    def __init__(self, exp_config, project_name):
        pass
    
    def log(self, dict):
        pass
    
    def finish(self):
        pass
    
class WandBTracker(Tracker):
    def __init__(self, exp_config, project_name):
        os.environ["WANDB_SILENT"] = "true"
        import wandb
        self.wandb = wandb
        wandb.init(
            config = exp_config,
            project = project_name)
    
    def log(self, dict):
        self.wandb.log(dict)
        
    def finish(self):
        self.wandb.finish()
        
class LocalTracker(Tracker):
    """
    Stores tracked data in a file.
    As writing to a file can be slow, this only writes the data when finish is called.
    """
    
    def __init__(self, exp_config, project_name):
        tracking_dir = config.LOCAL_TRACKING_PATH
        if not os.path.isdir(tracking_dir):
            os.mkdir(tracking_dir)
            
        self.setup_dict = {"config": vars(exp_config),
                           "project": project_name}
        file_name = str(datetime.datetime.now()).replace(" ", "_") + ".json"
        self.tracking_file_path = os.path.join(tracking_dir, file_name)
        
        print(self.setup_dict)
        # Write setup information to file in case we crash
        with open(self.tracking_file_path, 'w') as file:
            json.dump({"setup": self.setup_dict}, file)
            
        self.dicts = []
    
    def log(self, dict):
        self.dicts.append(dict)
    
    def finish(self):
        output_dict = {"setup": self.setup_dict,
                       "data": self.dicts}
        with open(self.tracking_file_path, 'w') as file:
            json.dump(output_dict, file)
            
def get_tracker(tracker_name, exp_config, project_name):
    match tracker_name:
        case "WandB":
            tracker = WandBTracker
        case "Local":
            tracker = LocalTracker
        case _:
            raise Exception("Unknown tracker name")
        
    return tracker(exp_config, project_name)
    