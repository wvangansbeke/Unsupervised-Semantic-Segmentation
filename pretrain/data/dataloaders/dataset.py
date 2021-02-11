#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch.utils.data as data
import random
import warnings

from copy import deepcopy
from torch.nn.functional import interpolate


class Dataset(data.Dataset):
    def __init__(self, base_dataset, train_transform, downsample_sal=False,
                    scale_factor_sal=0.125, min_area=0.01, max_area=0.99):
        super(Dataset, self).__init__()
        self.base_dataset = base_dataset
        self.train_transform = train_transform
        self.downsample_sal = downsample_sal
        
        if isinstance(scale_factor_sal, float):
            self.scale_factor_sal = (scale_factor_sal, scale_factor_sal)
        else:
            self.scale_factor_sal = scale_factor_sal

        self.min_area = min_area
        self.max_area = max_area

    def __len__(self):
        return len(self.base_dataset) 

    def __getitem__(self, index):
        sample_ = self.base_dataset.__getitem__(index)
        count = 0
        
        while True:
            if count > 1: # Warning
                #warnings.warn('Need to re-apply transform for image {}'.format(sample['meta']['image']))
                pass

            if count > 2: # Failed to load image two times in a row. Try a different one.
                #warnings.warn('Try loading a different image. Failed to load {}'.format(sample['meta']['image']))
                sample_ = self.base_dataset.__getitem__(random.randint(0, self.__len__()-1))
                count = 100
 
            sample = self.train_transform(deepcopy(sample_))
                           
            if self.downsample_sal: # Downsample
                sample['sal'] = interpolate(sample['sal'][None,None,:,:].float(),
                                            scale_factor=self.scale_factor_sal, mode='nearest').squeeze().long()
            area = sample['sal'].float().sum() / sample['sal'].numel()
            
            if area < self.max_area and area > self.min_area: # Ok. Foreground/Background has proper ratio.
                return sample

            else:
                count += 1 # Try again. Areas of foreground/background to small.


class DatasetKeyQuery(data.Dataset):
    def __init__(self, base_dataset, transform, downsample_sal=False,
                    scale_factor_sal=0.125, min_area=0.01, max_area=0.99):
        super(DatasetKeyQuery, self).__init__()
        self.base_dataset = base_dataset
        self.transform = transform
        self.downsample_sal = downsample_sal
        
        if isinstance(scale_factor_sal, float):
            self.scale_factor_sal = (scale_factor_sal, scale_factor_sal)
        else:
            self.scale_factor_sal = scale_factor_sal

        self.min_area = min_area
        self.max_area = max_area

    def __len__(self):
        return len(self.base_dataset) 

    def __getitem__(self, index):
        sample_ = self.base_dataset.__getitem__(index)
        count = 0
        
        while True:
            if count > 1: # Warning
                #warnings.warn('Need to re-apply transform for image {}'.format(sample['meta']['image']))
                pass

            if count > 2: # Failed to load image two times in a row. Try a different one.
                #warnings.warn('Try loading a different image. Failed to load {}'.format(sample['meta']['image']))
                sample_ = self.base_dataset.__getitem__(random.randint(0, self.__len__()-1))
                count = 100
 
            key_sample = self.transform(deepcopy(sample_))
            query_sample = self.transform(deepcopy(sample_))
                           
            if self.downsample_sal: # Downsample
                key_sample['sal'] = interpolate(key_sample['sal'][None,None,:,:].float(),
                                            scale_factor=self.scale_factor_sal, mode='nearest').squeeze().long()
                query_sample['sal'] = interpolate(query_sample['sal'][None,None,:,:].float(),
                                            scale_factor=self.scale_factor_sal, mode='nearest').squeeze().long()
            key_area = key_sample['sal'].float().sum() / key_sample['sal'].numel()
            query_area = query_sample['sal'].float().sum() / query_sample['sal'].numel()
            
            if key_area < self.max_area and key_area > self.min_area and query_area < self.max_area and query_area > self.min_area: # Ok. Foreground/Background has proper ratio.
                return {'key': key_sample, 'query': query_sample}

            else:
                count += 1 # Try again. Areas of foreground/background to small.


if __name__=='__main__':
    import numpy as np
    from matplotlib import pyplot as plt
    from utils.common_config import get_train_dataset, get_train_transformations
    p = {'train_db_name': 'VOCSegmentation', 'overfit': False}
    transform = get_train_transformations('strong')
    base_dataset = get_train_dataset(p, transform=None) 
    dataset = DatasetKeyQuery(base_dataset, transform, downsample_sal=False)

    for i, sample in enumerate(dataset):
        fig, axes = plt.subplots(4)
        key = np.transpose(sample['key']['image'].numpy(), (1,2,0))
        key = 255*(key * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
        query = np.transpose(sample['query']['image'].numpy(), (1,2,0))
        query = 255*(query * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
        sal_query = sample['query']['sal']
        sal_key = sample['key']['sal']
        axes[0].imshow(key.astype(np.uint8))
        axes[1].imshow(query.astype(np.uint8))
        axes[2].imshow(sal_key)
        axes[3].imshow(sal_query)
        plt.show()
