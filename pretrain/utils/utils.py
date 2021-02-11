#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import errno


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def freeze_layers(model):
    # Freeze block 1+2 layers in the backbone
    model.module.model_q.backbone.conv1.eval()
    model.module.model_q.backbone.bn1.eval()
    model.module.model_k.backbone.conv1.eval()
    model.module.model_k.backbone.bn1.eval()
    model.module.model_q.backbone.layer1.eval()
    model.module.model_k.backbone.layer1.eval()
    model.module.model_q.backbone.layer2.eval()
    model.module.model_k.backbone.layer2.eval()
    for name, param in model.module.model_q.backbone.conv1.named_parameters():
        param.requires_grad = False
    for name, param in model.module.model_q.backbone.bn1.named_parameters():
        param.requires_grad = False
    for name, param in model.module.model_k.backbone.conv1.named_parameters():
        param.requires_grad = False
    for name, param in model.module.model_k.backbone.bn1.named_parameters():
        param.requires_grad = False
    for name, param in model.module.model_q.backbone.layer1.named_parameters():
        param.requires_grad = False
    for name, param in model.module.model_q.backbone.layer2.named_parameters():
        param.requires_grad = False
    for name, param in model.module.model_k.backbone.layer1.named_parameters():
        param.requires_grad = False
    for name, param in model.module.model_k.backbone.layer2.named_parameters():
        param.requires_grad = False
    return model
