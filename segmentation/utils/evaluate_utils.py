#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utils.utils import SemsegMeter


"""
    Semantic segmentation evaluation
"""
@torch.no_grad()
def eval_segmentation_supervised_online(p, val_loader, model, verbose=True):
    """ Evaluate a segmentation network 
        The evaluation is performed online, without storing the results.
        
        Important: All images are assumed to be rescaled to the same resolution.
        As a consequence, the results might not exactly match with the true evaluation script
        if every image had a different size. 

        Alternative: Use store_results_to_disk and then evaluate with eval_segmentation_supervised_offline.
    """
    semseg_meter = SemsegMeter(p['num_classes'], val_loader.dataset.get_class_names(),
                            p['has_bg'], ignore_index=255)
    model.eval()

    for i, batch in enumerate(val_loader):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['semseg'].cuda(non_blocking=True)
        output = model(images)
        semseg_meter.update(torch.argmax(output, dim=1), targets)

    eval_results = semseg_meter.return_score(verbose = True)
    return eval_results


def eval_segmentation_supervised_offline(p, val_dataset, verbose=True):
    """ Evaluate stored predictions from a segmentation network.
        The semantic masks from the validation dataset are not supposed to change. 
    """
    n_classes = p['num_classes'] + int(p['has_bg'])

    # Iterate
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes
   
    for i, sample in enumerate(val_dataset):
        if i % 250 == 0:
            print('Evaluating: {} of {} objects'.format(i, len(val_dataset)))
        
        # Load result
        filename = os.path.join(p['save_dir'], sample['meta']['image'] + '.png')
        mask = np.array(Image.open(filename)).astype(np.uint8)

        gt = sample['semseg']
        valid = (gt != 255)

        if mask.shape != gt.shape:
            warning.warn('Prediction and ground truth have different size. Resizing Prediction ..')
            mask = cv2.resize(mask, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)

        # TP, FP, and FN evaluation
        for i_part in range(0, n_classes):
            tmp_gt = (gt == i_part)
            tmp_pred = (mask == i_part)
            tp[i_part] += np.sum(tmp_gt & tmp_pred & valid)
            fp[i_part] += np.sum(~tmp_gt & tmp_pred & valid)
            fn[i_part] += np.sum(tmp_gt & ~tmp_pred & valid)

    jac = [0] * n_classes
    for i_part in range(0, n_classes):
        jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

    # Write results
    eval_result = dict()
    eval_result['jaccards_all_categs'] = jac
    eval_result['mIoU'] = np.mean(jac)
        
    if verbose:
        print('Evaluation of semantic segmentation ')
        print('mIoU is %.2f' %(100*eval_result['mIoU']))
        class_names = val_dataset.get_class_names()
        for i_part in range(n_classes):
            print('IoU class %s is %.2f' %(class_names[i_part], 100*jac[i_part]))

    return eval_result


@torch.no_grad()
def save_results_to_disk(p, val_loader, model, crf_postprocess=False):
    print('Save results to disk ...')
    model.eval()

    # CRF
    if crf_postprocess:
        from utils.crf import dense_crf

    counter = 0
    for i, batch in enumerate(val_loader):
        output = model(batch['image'].cuda(non_blocking=True))
        meta = batch['meta']
        for jj in range(output.shape[0]):
            counter += 1
            image_file = meta['image_file'][jj]

            # CRF post-process
            if crf_postprocess:
                probs = dense_crf(meta['image_file'][jj], output[jj])
                pred = np.argmax(probs, axis=0).astype(np.uint8)
            
            # Regular
            else:
                pred = torch.argmax(output[jj], dim=0).cpu().numpy().astype(np.uint8)

            result = cv2.resize(pred, dsize=(meta['im_size'][1][jj], meta['im_size'][0][jj]), 
                                        interpolation=cv2.INTER_NEAREST)
            imageio.imwrite(os.path.join(p['save_dir'], meta['image'][jj] + '.png'), result)
   
        if counter % 250 == 0:
            print('Saving results: {} of {} objects'.format(counter, len(val_loader.dataset)))
