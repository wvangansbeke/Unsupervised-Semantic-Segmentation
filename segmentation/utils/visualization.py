#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def visualize_sample(sample, filename=None):
    # Visualize a sample and its segmentation map
    image = sample['image']
    meta = sample['meta']
    im_size = sample['meta']['im_size']

    image[0] = image[0] * 0.229 + 0.485
    image[1] = image[1] * 0.224 + 0.456
    image[2] = image[2] * 0.225 + 0.406
    image = np.transpose(255*image.numpy(), (1,2,0)).astype(np.uint8)
    image = cv2.resize(image, im_size, interpolation=cv2.INTER_CUBIC)
    

    if 'semseg' in sample.keys():
        sem = sample['semseg'].squeeze().numpy().astype(np.uint8)
        sem = cv2.resize(sem, im_size, interpolation=cv2.INTER_NEAREST)
        cmap = color_map()
        array = np.empty((sem.shape[0], sem.shape[1], cmap.shape[1]), dtype=cmap.dtype)
        for class_i in np.unique(sem):
            array[sem == class_i]  = cmap[class_i]
        plt.figure()
        plt.subplot(1,2,1)
        plt.axis('off')
        plt.imshow(image)
        plt.title('RGB')
        plt.subplot(1,2,2)
        plt.imshow(image)
        plt.imshow(array, alpha=0.6)
        plt.axis('off')
        plt.title('GT')
    
    else:
        fig = plt.imshow(image)
        plt.title('RGB')
        plt.axis('off')

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)
   
    else:
        plt.show()


def visualize_sample_with_saved_prediction(p, sample, filename):
    # Visualize a sample with ground-truth and saved prediction
    # Input should come from the true validation dataset without any transformations being applied

    # Read sample
    image = sample['image']
    meta = sample['meta']
    sem = sample['semseg']
    cmap = color_map()

    # Apply color map to visualize segmentation ground-truth
    array = np.empty((sem.shape[0], sem.shape[1], cmap.shape[1]), dtype=cmap.dtype)
    for class_i in np.unique(sem):
        array[sem == class_i]  = cmap[class_i]

    # Read and apply color map to visualize segmentation prediction
    mask = (os.path.join(p['save_dir'], meta['image'] + '.png'))
    mask = np.array(Image.open(mask)).astype(np.uint8)
    assert(mask.shape[0] == sem.shape[0] and mask.shape[1] == sem.shape[1])
    array_pred = np.empty((mask.shape[0], mask.shape[1], cmap.shape[1]), dtype=cmap.dtype)
    for class_i in np.unique(mask):
        array_pred[mask == class_i]  = cmap[class_i]

    plt.figure()

    plt.subplot(1,3,1)
    plt.axis('off')
    plt.imshow(image)
    plt.title('RGB')

    plt.subplot(1,3,2)
    plt.imshow(image)
    plt.imshow(array, alpha=0.6)
    plt.axis('off')
    plt.title('GT')

    plt.subplot(1,3,3)
    plt.imshow(image)
    plt.imshow(array_pred, alpha=0.6)
    plt.axis('off')
    plt.title('Pred')

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)
   
    else:
        plt.show()


def visualize_sample_with_prediction(image, gt, prediction, filename=None):
    # Visualize a sample, gt and the prediction
    cmap = color_map()

    # Image
    image = image.cpu().numpy()
    image[0] = image[0] * 0.229 + 0.485
    image[1] = image[1] * 0.224 + 0.456
    image[2] = image[2] * 0.225 + 0.406
    image = np.transpose(255*image, (1,2,0)).astype(np.uint8)
    
    # Semantic gt
    gt = gt.cpu().numpy().astype(np.uint8) 
    array_gt = np.empty((gt.shape[0], gt.shape[1], cmap.shape[1]), dtype=cmap.dtype)
    for class_i in np.unique(gt):
        array_gt[gt == class_i]  = cmap[class_i]

    # Prediction
    prediction = prediction.cpu().numpy().astype(np.uint8) 
    array_pred = np.empty((prediction.shape[0], prediction.shape[1], cmap.shape[1]), dtype=cmap.dtype)
    for class_i in np.unique(prediction):
        array_pred[prediction == class_i]  = cmap[class_i]
    
    fig, axes = plt.subplots(3)
    axes[0].imshow(image)
    axes[1].imshow(array_gt)
    axes[2].imshow(array_pred)
    plt.axis('off')
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)
   
    else:
        plt.show()

 

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap
