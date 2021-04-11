#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import numpy as np
import sys
import torch
import torch.nn as nn

from utils.config import create_config
from utils.common_config import get_val_dataset, get_val_transformations,\
                                get_val_dataloader,\
                                get_model
from utils.logger import Logger
from utils.retrieval_utils import build_memory_bank, retrieval
from termcolor import colored


# Parser
parser = argparse.ArgumentParser(description='Retrieval')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()


def main():
    cv2.setNumThreads(1)

    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    sys.stdout = Logger(os.path.join(p['retrieval_dir'], 'log_file.txt'))
    print('Python script is {}'.format(os.path.abspath(__file__)))
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print(model)
    model = model.cuda()

    # Load pre-trained weights
    state_dict = torch.load(p['pretraining'], map_location='cpu')
        # State dict follows our lay-out
    if 'model' in state_dict.keys():
        state_dict = state_dict['model']
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('module.model_q'):
            new_state[k.rsplit('module.model_q.')[1]] = v
    msg = model.load_state_dict(new_state, strict=False)
    print(msg)

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    
    # Transforms 
    from data.dataloaders.pascal_voc import VOC12
    val_transforms = get_val_transformations()
    print(val_transforms)
    train_dataset = VOC12(split='train', transform=val_transforms, ignore_classes=p['retrieval_kwargs']['ignore_classes'])
    val_dataset = VOC12(split='val', transform=val_transforms, ignore_classes=p['retrieval_kwargs']['ignore_classes'])
    train_dataloader = get_val_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train dataset {} - Val dataset {}'.format(str(train_dataset), str(val_dataset)))
    print('Train samples {} - Val samples {}' .format(len(train_dataset), len(val_dataset)))

    # Build memory bank
    print(colored('Perform retrieval ...', 'blue'))
    memory_bank = build_memory_bank(p, train_dataset, train_dataloader, model)
    results = retrieval(p, memory_bank, val_dataset, val_dataloader, model)
    

if __name__ == "__main__":
    main()

