#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import numpy as np
import torch
import torch.nn as nn

from utils.utils import SemsegMeter

@torch.no_grad()
def build_memory_bank(p, dataset, loader, model):
    print('Building memory bank ...') 
    
    model.eval()
    ptr = 0
    
    all_prototypes = torch.zeros(len(dataset), p['model_kwargs']['ndim']).float()
    all_labels = torch.zeros(len(dataset)).long()
    
    for i, batch in enumerate(loader):
        semseg = batch['semseg']
        output, sal = model(batch['image'].cuda(non_blocking=True))

        # compute prototype per salient object
        bs, dim, _, _ = output.shape
        output = output.reshape(bs, dim, -1) # B x dim x H.W
        sal_proto = sal.reshape(bs, -1, 1).type(output.dtype) # B x H.W x 1
        prototypes = torch.bmm(output, sal_proto*(sal_proto>0.5).float()).squeeze() # B x dim
        prototypes = nn.functional.normalize(prototypes, dim=1)

        # compute majority vote per salient object
        sal = (sal > 0.5).cpu()
        for jj in range(bs):
            sal_jj, semseg_jj = sal[jj], semseg[jj]

            # did we detect a salient object
            if torch.sum(sal_jj).item() == 0:
                continue
            
            # does the salient object contain a class of interest - Not all background/ignore index.
            # suggestion for improvement -> throw out image based upon gt.
            classes, counts = np.unique(semseg_jj[sal_jj].numpy(), return_counts=True)
            if set(classes).issubset(set({0,255})):
                continue

            # get majority vote
            majority_vote = max([(count_, class_) for class_, count_ in zip(classes, counts) if class_ not in [0,255]])[1]
            all_prototypes[ptr] = prototypes[jj]
            all_labels[ptr] = majority_vote
            ptr += 1

        # print progress
        if (i + 1) % 25 == 0:
            print('Progress [{}/{}]'.format(i+1, len(loader)))

    return {'prototypes': all_prototypes[:ptr], 'labels': all_labels[:ptr]}


@torch.no_grad()
def retrieval(p, memory_bank, val_dataset, val_loader, model):
    print('Performing retrieval ...')
    model.eval()

    memory_prototypes = memory_bank['prototypes'].cuda()
    memory_labels = memory_bank['labels'].cuda()

    meter = SemsegMeter(p['num_classes'], val_dataset.get_class_names(),
                            p['has_bg'], ignore_index=255)

    for i, batch in enumerate(val_loader):
        semseg = batch['semseg'].cuda(non_blocking=True)
        b, h, w = semseg.size()
        output, sal = model(batch['image'].cuda(non_blocking=True))

        # compute prototype per salient object
        bs, dim, _, _ = output.shape
        output = output.reshape(bs, dim, -1) # B x dim x H.W
        sal_proto = sal.reshape(bs, -1, 1).type(output.dtype) # B x H.W x 1
        prototypes = torch.bmm(output, sal_proto*(sal_proto>0.5).float()).squeeze() # B x dim
        prototypes = nn.functional.normalize(prototypes, dim=1)

        # find nearest neighbor
        correlation = torch.matmul(prototypes, memory_prototypes.t())
        neighbors = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(memory_labels, 0, neighbors)

        # construct prediction
        pred = torch.LongTensor(b, h, w).zero_().cuda()
        for jj in range(b):
            pred[jj][sal[jj] > 0.5] = class_pred[jj]

        # update meter
        meter.update(pred, semseg)
            
        # print progress
        if (i + 1) % 25 == 0:
            print('Progress [{}/{}]'.format(i+1, len(val_loader)))

    
    if len(val_dataset.ignore_classes) == 0: # We keep all classes
        eval_results = meter.return_score(verbose=True)

    else: # We defined classes to be ignored - Also remove background
        print('Evaluation of semantic segmentation')
        eval_results = meter.return_score(verbose=False)
        ignore_classes = [0] + val_dataset.ignore_classes 
        iou = []
        for i in range(p['num_classes'] + p['has_bg']):
            if i in ignore_classes:
                continue
            
            print('IoU class {} is {:.2f}'.format(val_dataset.get_class_names()[i], 100*eval_results['jaccards_all_categs'][i]))
            iou.append(eval_results['jaccards_all_categs'][i])
        print('Mean IoU is {:.2f}'.format(100*np.mean(iou)))
