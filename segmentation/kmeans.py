#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import numpy as np
import torch
import torch.nn as nn

from utils.config import create_config
from utils.common_config import get_val_dataset, get_val_transformations,\
                                get_val_dataloader,\
                                get_model
from utils.kmeans_utils import save_embeddings_to_disk, eval_kmeans
from termcolor import colored
import torchvision.transforms as transforms
from termcolor import colored

# Parser
parser = argparse.ArgumentParser(description='Fully-supervised segmentation')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--num_seeds', default=5, type=int,
                    help='number of seeds during kmeans')
args = parser.parse_args()

def main():
    cv2.setNumThreads(1)

    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
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
    val_dataset = VOC12(split='val', transform=val_transforms)
    val_dataloader = get_val_dataloader(p, val_dataset)

    true_val_dataset = VOC12(split='val', transform=None)
    print(colored('Val samples %d' %(len(true_val_dataset)), 'yellow'))

    # Kmeans Clustering
    n_clusters = 21
    results_miou = []
    for i in range(args.num_seeds):
        save_embeddings_to_disk(p, val_dataloader, model, n_clusters=n_clusters, seed=1234 + i)
        eval_stats = eval_kmeans(p, true_val_dataset, n_clusters=n_clusters, verbose=True)
        results_miou.append(eval_stats['mIoU'])
    print(colored('Average mIoU is %2.1f' %(np.mean(results_miou)*100), 'green'))
    

if __name__ == "__main__":
    main()
