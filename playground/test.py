from act_jian.detr.models.CVAE_clip import CVAE

import yaml
config_path = '/home/jian/git_all/git_manipulation/act_jian/act_jian/jianact.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
model = CVAE(config)
for name, param in model.named_parameters():
    print(f"Parameter: {name}, Precision: {param.dtype}")
