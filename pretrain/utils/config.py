#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing


def load_config(config_file_exp):
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()

    for k, v in config.items():
        cfg[k] = v

    return cfg


def create_config(config_file_env, config_file_exp):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']
   
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()
   
    # Copy
    for k, v in config.items():
        cfg[k] = v

    # Set paths
    output_dir = os.path.join(root_dir, os.path.basename(config_file_exp).split('.')[0])
    mkdir_if_missing(output_dir)
    cfg['output_dir'] = output_dir
    cfg['checkpoint'] = os.path.join(cfg['output_dir'], 'checkpoint.pth.tar')
    cfg['best_model'] = os.path.join(cfg['output_dir'], 'best_model.pth.tar')

    return cfg 
