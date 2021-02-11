#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import numpy.random as random
import numpy as np
import torch
import torchvision
from PIL import Image
import torchvision.transforms.functional as F


class RandomResizedCrop(torchvision.transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        super(RandomResizedCrop, self).__init__(size, scale=scale, ratio=ratio)
        self.interpolation_img = Image.BILINEAR
        self.interpolation_sal = Image.NEAREST
    
    def __call__(self, sample):
        img = sample['image']
        sal = sample['sal']

        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        sample['image'] = F.resized_crop(img, i, j, h, w, self.size, self.interpolation_img)
        sample['sal'] = F.resized_crop(sal, i, j, h, w, self.size, self.interpolation_sal)
        return sample


class Resize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise ValueError('Invalid type size {}'.format(type(size)))
    
        self.resize_img = torchvision.transforms.Resize(self.size, interpolation=Image.BILINEAR)
        self.resize_sal = torchvision.transforms.Resize(self.size, interpolation=Image.NEAREST)

    def __call__(self, sample):
        sample['image'] = self.resize_img(sample['image'])
        sample['sal'] = self.resize_sal(sample['sal'])
        return sample


class ColorJitter(object):
    def __init__(self, jitter):
        self.jitter = torchvision.transforms.ColorJitter(jitter[0], jitter[1], jitter[2], jitter[3])

    def __call__(self, sample):
        sample['image'] = self.jitter(sample['image']) 
        return sample

    def __str__(self):
        return 'ColorJitter'


class RandomHorizontalFlip(object):
    def __call__(self, sample):

        if random.random() < 0.5:
            sample['image'] = sample['image'].transpose(Image.FLIP_LEFT_RIGHT)
            sample['sal'] = sample['sal'].transpose(Image.FLIP_LEFT_RIGHT)

        return sample
    
    def __str__(self):
        return 'RandomHorizontalFlip'


class RandomGrayscale(object):
    def __init__(self, p=0.2):
        self.p = p
    
    def __call__(self, sample):
        img = sample['image']
        num_output_channels = 1 if img.mode == 'L' else 3
        if random.random() < self.p:
            sample['image'] = F.to_grayscale(img, num_output_channels=num_output_channels)
        return sample

    def __str__(self):
        return 'RandomGrayscale'

class ToTensor(object):
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, sample):
        sample['image'] = self.to_tensor(sample['image'])
        sal_ = self.to_tensor(sample['sal']).squeeze().long()
        if len(sal_.shape) == 3:
            sample['sal'] = sal_[0]
        else:
            sample['sal'] = sal_

        return sample
    
    def __str__(self):
        return 'ToTensor'


class Normalize(object):
    def __init__(self, mean, std):
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        sample['image'] = self.normalize(sample['image'])
        return sample

    def __str__(self):
        return 'Normalize'
