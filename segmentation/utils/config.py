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


def create_config(config_file_env, config_file_exp, run_idx=None):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']
   
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()
   
    # Copy
    for k, v in config.items():
        cfg[k] = v

    # Num classes
    if cfg['train_db_name'] == 'VOCSegmentation':
        cfg['num_classes'] = 20
        cfg['has_bg'] = True
    
    else:
        raise ValueError('Invalid train db name {}'.format(cfg['train_db_name']))
  
    # Paths 
    output_dir = os.path.join(root_dir, os.path.basename(config_file_exp).split('.')[0])
    mkdir_if_missing(output_dir)
    
    cfg['output_dir'] = output_dir
    cfg['checkpoint'] = os.path.join(cfg['output_dir'], 'checkpoint.pth.tar')
    cfg['best_model'] = os.path.join(cfg['output_dir'], 'best_model.pth.tar')
    cfg['save_dir'] = os.path.join(cfg['output_dir'], 'predictions')
    
    mkdir_if_missing(cfg['save_dir'])
    cfg['log_file'] = os.path.join(cfg['output_dir'], 'logger.txt')

    # Special directories for K-Means -> Which happens off-line
    cfg['embedding_dir'] = os.path.join(cfg['output_dir'], 'embeddings')
    cfg['sal_dir'] = os.path.join(cfg['output_dir'], 'saliency')
    mkdir_if_missing(cfg['embedding_dir'])
    mkdir_if_missing(cfg['sal_dir'])

    # Special directories for retrieval
    cfg['retrieval_dir'] = os.path.join(cfg['output_dir'], 'retrieval')
    mkdir_if_missing(cfg['retrieval_dir'])
    
    if 'kmeans_eval' not in cfg.keys():
        cfg['kmeans_eval'] = False

    return cfg 
