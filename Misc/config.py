import yaml
from Misc.utils import dotdict

config = dotdict(yaml.safe_load(open("./Configs/config.yaml")))

