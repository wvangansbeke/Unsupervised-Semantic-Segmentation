#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import math
import numpy as np
import torch
import torchvision
import data.dataloaders.transforms as transforms
from data.util.mypath import Path
from utils.collate import collate_custom


def load_pretrained_weights(p, model):
    if p['backbone_kwargs']['pretraining'] == 'imagenet_classification':
        # Load pre-trained ImageNet classification weights from torchvision
        from torchvision.models.utils import load_state_dict_from_url
        print('Loading ImageNet pre-trained classification (supervised) weights for backbone')
        model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        }
        state_dict = load_state_dict_from_url(model_urls[p['backbone']],
                                              progress=True)
        msg = model.load_state_dict(state_dict, strict=False)        
        print(msg)

    elif p['backbone_kwargs']['pretraining'] == 'imagenet_moco':
        # Load pre-trained ImageNet MoCo weights
        print('Loading MoCo pre-trained weights for backbone')
        state_dict = torch.load(p['backbone_kwargs']['moco_state_dict'], map_location='cpu')['state_dict']
        new_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.encoder_q'):
                new_dict[k.rsplit('module.encoder_q.')[1]] = v
        msg = model.load_state_dict(new_dict, strict=False)   
        assert(all(['fc' in k for k in msg[0]])) 
        assert(all(['fc' in k for k in msg[1]])) 

    else:
        raise ValueError('Invalid value {}'.format(p['backbone_kwargs']['pretraining']))
 
def get_model(p):
    # Get backbone
    if p['backbone'] == 'resnet18':
        import torchvision.models.resnet as resnet
        backbone = resnet.__dict__['resnet18'](pretrained=False)
        backbone_channels = 512
    
    elif p['backbone'] == 'resnet50':
        import torchvision.models.resnet as resnet
        backbone = resnet.__dict__['resnet50'](pretrained=False)
        backbone_channels = 2048

    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    # Load pretrained weights
    if p['backbone_kwargs']['pretraining']:
        load_pretrained_weights(p, backbone)

    if p['backbone_kwargs']['dilated']:
        from modules.resnet_dilated import ResnetDilated
        backbone = ResnetDilated(backbone)
    
    # Get head
    if p['head'] == 'deeplab':
        from modules.deeplab import DeepLabHead
        nc = p['model_kwargs']['ndim']
        head = DeepLabHead(backbone_channels, nc)

    else:
        raise ValueError('Invalid head {}'.format(p['head']))

    # Compose model from backbone and head
    from modules.models import ContrastiveSegmentationModel
    return ContrastiveSegmentationModel(backbone, head, p['model_kwargs']['head'], 
                                                p['model_kwargs']['upsample'], 
                                                p['model_kwargs']['use_classification_head'])


def get_train_dataset(p, transform=None):
    if p['train_db_name'] == 'VOCSegmentation':
        from data.dataloaders.pascal_voc import VOCSegmentation
        return VOCSegmentation(root=Path.db_root_dir(p['train_db_name']),
                            saliency=p['train_db_kwargs']['saliency'],
                            transform=transform)
    
    else:    
        raise ValueError('Invalid train db name {}'.format(p['train_db_name']))   
 

def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['train_batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=True, shuffle=True)


def get_train_transformations():
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        torchvision.transforms.RandomApply([
            transforms.ColorJitter([0.4, 0.4, 0.4, 0.1])
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ]

    return torchvision.transforms.Compose(augmentation)

    
def get_val_transformations():
    augmentation = [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ]
        
    return torchvision.transforms.Compose(augmentation)


def get_optimizer(p, parameters):
    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(parameters, **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'poly':
        lambd = pow(1-(epoch/p['epochs']), 0.9)
        lr = lr * lambd

    elif p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
