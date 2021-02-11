#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import sys
import errno
import hashlib
import glob
import tarfile
import numpy as np
import torch.utils.data as data

from data.util.mypath import Path
from data.util.google_drive import download_file_from_google_drive
from utils.utils import mkdir_if_missing
from PIL import Image


class VOCSegmentation(data.Dataset):
    
    GOOGLE_DRIVE_ID = '1pxhY5vsLwXuz6UHZVUKhtb7EJdCg2kuH'

    FILE = 'PASCAL_VOC.tgz'

    def __init__(self, root=Path.db_root_dir('VOCSegmentation'),
                 saliency='supervised_model', download=True,
                 transform=None, overfit=False):
        super(VOCSegmentation, self).__init__()

        self.root = root
        self.transform = transform

        if download:
            self._download()
        
        self.images_dir = os.path.join(self.root, 'images')
        valid_saliency = ['supervised_model', 'unsupervised_model']
        assert(saliency in valid_saliency)
        self.saliency = saliency
        self.sal_dir = os.path.join(self.root, 'saliency_' + self.saliency)
    
        self.images = []
        self.sal = []

        with open(os.path.join(self.root, 'sets/trainaug.txt'), 'r') as f:
            all_ = f.read().splitlines()

        for f in all_:
            _image = os.path.join(self.images_dir, f + ".jpg")
            _sal = os.path.join(self.sal_dir, f + ".png")
            if os.path.isfile(_image) and os.path.isfile(_sal):
                self.images.append(_image)
                self.sal.append(_sal)

        assert (len(self.images) == len(self.sal))

        if overfit:
            n_of = 32
            self.images = self.images[:n_of]
            self.sal = self.sal[:n_of]

        # Display stats
        print('Number of images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}

        sample['image'] = self._load_img(index)
        sample['sal'] = self._load_sal(index)

        if self.transform is not None:
            sample = self.transform(sample)
        
        sample['meta'] = {'image': str(self.images[index])}

        return sample

    def __len__(self):
            return len(self.images)

    def _load_img(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        return _img

    def _load_sal(self, index):
        _sal = Image.open(self.sal[index])
        return _sal

    def __str__(self):
        return 'VOCSegmentation(saliency=' + self.saliency + ')'

    def get_class_names(self):
        # Class names for sal
        return ['background', 'salient object']
    
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
    # Sample from supervised saliency model
    dataset = VOCSegmentation(saliency='supervised_model')
    sample = dataset.__getitem__(5)
    fig, axes = plt.subplots(2)
    axes[0].imshow(sample['image'])
    axes[1].imshow(sample['sal'])
    plt.show()
    plt.close()

    # Sample from unsupervised saliency model
    dataset = VOCSegmentation(saliency='unsupervised_model')
    sample = dataset.__getitem__(5)
    fig, axes = plt.subplots(2)
    axes[0].imshow(sample['image'])
    axes[1].imshow(sample['sal'])
    plt.show()
    plt.close()
