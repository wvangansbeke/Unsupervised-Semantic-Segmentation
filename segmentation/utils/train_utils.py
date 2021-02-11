#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
from utils.utils import AverageMeter, ProgressMeter
from utils.utils import SemsegMeter


def train_segmentation_vanilla(p, train_loader, model, criterion, optimizer, epoch, freeze_batchnorm='none'):
    """ Train a segmentation model in a fully-supervised manner """
    losses = AverageMeter('Loss', ':.4e')
    semseg_meter = SemsegMeter(p['num_classes'], train_loader.dataset.get_class_names(),
                            p['has_bg'], ignore_index=255)
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    if freeze_batchnorm == 'none':
        print('BatchNorm tracks running stats - model put to train mode.')
        pass
    
    elif freeze_batchnorm == 'backbone':
        print('Freeze BatchNorm in the backbone - backbone put to eval mode.')
        model.backbone.eval() # Put encoder to eval

    elif freeze_batchnorm == 'all': # Put complete model to eval
        print('Freeze BatchNorm - model put to eval mode.')
        model.eval()

    else:
        raise ValueError('Invalid value freeze batchnorm {}'.format(freeze_batchnorm))

    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['semseg'].cuda(non_blocking=True)

        output = model(images)
        loss = criterion(output, targets)
        losses.update(loss.item())
        semseg_meter.update(torch.argmax(output, dim=1), targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            progress.display(i)

    eval_results = semseg_meter.return_score(verbose = True)
    return eval_results
