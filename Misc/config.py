import yaml
from Misc.utils import dotdict

config_dict = yaml.safe_load(open("./Configs/config.yaml"))

if not config_dict["tracker"] in ["Local", "WandB", "Comet", "None"]:
    raise Exception("Tracking parameter in config is set to a not supported value")

config_dict["use_tracking"] = config_dict["tracker"] != "None"

config = dotdict(config_dict)
