#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import sys
import errno
import cv2
import hashlib
import glob
import tarfile

import numpy as np
import torch.utils.data as data
import torch
from PIL import Image

from data.util.mypath import Path
from data.util.google_drive import download_file_from_google_drive
from utils.utils import mkdir_if_missing


class VOC12(data.Dataset):
    
    GOOGLE_DRIVE_ID = '1pxhY5vsLwXuz6UHZVUKhtb7EJdCg2kuH'

    FILE = 'PASCAL_VOC.tgz'

    VOC_CATEGORY_NAMES = ['background',
                          'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self, root=Path.db_root_dir('VOCSegmentation'),
                 split='val', transform=None, download=True):
        # Set paths
        self.root = root
        valid_splits = ['trainaug', 'train', 'val']
        assert(split in valid_splits)
        self.split = split
         
        if split == 'trainaug':
            _semseg_dir = os.path.join(self.root, 'SegmentationClassAug')
        else:
            _semseg_dir = os.path.join(self.root, 'SegmentationClass')

        _image_dir = os.path.join(self.root, 'images')


        # Download
        if download:
            self._download()

        # Transform
        self.transform = transform

        # Splits are pre-cut
        print("Initializing dataloader for PASCAL VOC12 {} set".format(''.join(self.split)))
        split_file = os.path.join(self.root, 'sets', self.split + '.txt')
        self.images = []
        self.semsegs = []
        
        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            # Images
            _image = os.path.join(_image_dir, line + ".jpg")
            assert os.path.isfile(_image)
            self.images.append(_image)

            # Semantic Segmentation
            _semseg = os.path.join(_semseg_dir, line + '.png')
            assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

        assert(len(self.images) == len(self.semsegs))

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}

        # Load image
        _img = self._load_img(index)
        sample['image'] = _img

        # Load pixel-level annotations
        _semseg = self._load_semseg(index)
        if _semseg.shape != _img.shape[:2]:
            _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        sample['semseg'] = _semseg
	
        sample['meta'] = {'im_size': (_img.shape[0], _img.shape[1]),
                          'image_file': self.images[index],
                          'image': os.path.basename(self.semsegs[index]).split('.')[0]}
            
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB'))
        return _img

    def _load_semseg(self, index):
        _semseg = np.array(Image.open(self.semsegs[index]))
        return _semseg

    def get_img_size(self, idx=0):
        img = Image.open(os.path.join(self.root, 'JPEGImages', self.images[idx] + '.jpg'))
        return list(reversed(img.size))

    def __str__(self):
        return 'VOC12(split=' + str(self.split) + ')' 

    def get_class_names(self):
        return self.VOC_CATEGORY_NAMES

    def _download(self):
        _fpath = os.path.join(Path.db_root_dir(), self.FILE)

        if os.path.isfile(_fpath):
            print('Files already downloaded')
            return
        else:
            print('Downloading dataset from google drive')
            mkdir_if_missing(os.path.dirname(_fpath))
            download_file_from_google_drive(self.GOOGLE_DRIVE_ID, _fpath)

        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(Path.db_root_dir())
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')


if __name__ == '__main__':
    """ For purpose of debugging """
    from matplotlib import pyplot as plt
    dataset = VOC12(split='train', transform=None)

    fig, axes = plt.subplots(2)
    sample = dataset.__getitem__(0)
    axes[0].imshow(sample['image'])
    axes[1].imshow(sample['semseg'])
    plt.show()
