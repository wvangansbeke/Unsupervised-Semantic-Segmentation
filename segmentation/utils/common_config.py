#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import data.dataloaders.custom_transforms as custom_tr
from utils.collate import collate_custom

 
def load_pretrained_weights(p, model):
    # Load weights from pre-training
    print('Loading pre-trained weights from {}'.format(p['pretraining']))
    state_dict = torch.load(p['pretraining'], map_location='cpu')['model']
    new_state = {}

    for k, v in state_dict.items():
        if k.startswith('module.model_q.'):
            new_state[k.rsplit('module.model_q.')[1]] = v
        
        else:
            pass

    msg = model.load_state_dict(new_state, strict=False)
    print('Loading state dict from checkpoint')
    print('Warning: This piece of code was only tested for linear classification')
    print('Warning: Assertions should probably depend on model type (Segm/ContrastiveSegm)')
    assert(set(msg[0]) == set(['decoder.4.weight', 'decoder.4.bias']))
    assert(set(msg[1]) == set(['head.weight', 'head.bias', 'classification_head.weight'])) 
    
    # Init final conv layer
    if 'deeplab' in p['head']:
        model.decoder[4].weight.data.normal_(mean=0.0, std=0.01)
        model.decoder[4].bias.data.zero_()


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

    if p['backbone_kwargs']['dilated']:
        from models.resnet_dilated import ResnetDilated
        backbone = ResnetDilated(backbone)
    
    # Get head
    if p['head'] == 'deeplab':
        if not p['kmeans_eval']:
            nc = p['num_classes'] + int(p['has_bg'])
        else:
            nc = p['model_kwargs']['ndim']

        from models.deeplab import DeepLabHead
        head = DeepLabHead(backbone_channels, nc)

    elif p['head'] == 'dim_reduction':
        nc = p['num_classes'] + int(p['has_bg'])
        import torch.nn as nn
        head = nn.Conv2d(backbone_channels, nc, 1)

    else:
        raise ValueError('Invalid head {}'.format(p['head']))

    # Compose model from backbone and head
    if p['kmeans_eval']:
        from models.models import ContrastiveSegmentationModel
        import torch.nn as nn
        model = ContrastiveSegmentationModel(backbone, head, p['model_kwargs']['head'], 
                                                    p['model_kwargs']['upsample'], 
                                                    p['model_kwargs']['use_classification_head'], p['freeze_layer'])
    else:
        from models.models import SimpleSegmentationModel
        model = SimpleSegmentationModel(backbone, head)
    
        # Load pretrained weights
        load_pretrained_weights(p, model)
    return model


def get_train_dataset(p, transform=None):
    if p['train_db_name'] == 'VOCSegmentation':
        from data.dataloaders.pascal_voc import VOC12
        dataset = VOC12(split=p['train_db_kwargs']['split'], transform=transform)
    
    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
    
    return dataset


def get_val_dataset(p, transform=None):
    if p['val_db_name'] == 'VOCSegmentation':
        from data.dataloaders.pascal_voc import VOC12
        dataset = VOC12(split='val', transform=transform)        
    
    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))
    
    return dataset


def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['train_db_kwargs']['batch_size'], pin_memory=True, 
            collate_fn=collate_custom, drop_last=True, shuffle=True)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['val_db_kwargs']['batch_size'], pin_memory=True, 
            collate_fn=collate_custom, drop_last=False, shuffle=False)


def get_train_transformations(augmentation_strategy='pascal'):
    return transforms.Compose([custom_tr.RandomHorizontalFlip(),
                                   custom_tr.ScaleNRotate(rots=(-5,5), scales=(.75,1.25),
                                    flagvals={'semseg': cv2.INTER_NEAREST, 'image': cv2.INTER_CUBIC}),
                                   custom_tr.FixedResize(resolutions={'image': tuple((512,512)), 'semseg': tuple((512,512))},
                                    flagvals={'semseg': cv2.INTER_NEAREST, 'image': cv2.INTER_CUBIC}),
                                   custom_tr.ToTensor(),
                                    custom_tr.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    
def get_val_transformations():
    return transforms.Compose([custom_tr.FixedResize(resolutions={'image': tuple((512,512)), 
                                                        'semseg': tuple((512,512))},
                                            flagvals={'image': cv2.INTER_CUBIC, 'semseg': cv2.INTER_NEAREST}),
                                custom_tr.ToTensor(),
                                custom_tr.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


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

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
