#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import builtins
import os
import sys
from termcolor import colored

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from data.dataloaders.dataset import DatasetKeyQuery

from modules.moco.builder import ContrastiveModel

from utils.config import create_config
from utils.common_config import get_train_dataset, get_train_transformations,\
                                get_train_dataloader, get_optimizer, adjust_learning_rate

from utils.train_utils import train
from utils.logger import Logger
from utils.collate import collate_custom


# Parser
parser = argparse.ArgumentParser(description='Main function')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--nvidia-apex', action='store_true',
                    help='Use mixed precision')

# Distributed params
parser.add_argument('--world-size', default=1, type=int,
                            help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                            help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                 'N processes per node, which has N GPUs. This is the '
                                 'fastest way to use PyTorch for either single node or '
                                 'multi node data parallel training')

def main():
    args = parser.parse_args()
    args.multiprocessing_distributed = True
    assert args.multiprocessing_distributed
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    # main_worker(0, ngpus_per_node, args=args)


def main_worker(gpu, ngpus_per_node, args):
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)

    # Check gpu id
    args.gpu = gpu
    p['gpu'] = gpu
    if args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    else:
        sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'))
        
    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])

    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                    world_size=args.world_size, rank=args.rank)

    print('Python script is {}'.format(os.path.abspath(__file__)))
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = ContrastiveModel(p)
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    
    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model.parameters())
    print(optimizer)

    # Nvidia-apex
    if args.nvidia_apex:
        print(colored('Using mixed precision training', 'blue'))
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2",
                                            keep_batchnorm_fp32=True, loss_scale="dynamic")
    else:
        amp = None
    
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    p['train_batch_size'] = int(p['train_batch_size'] / ngpus_per_node)
    p['num_workers'] = int((p['num_workers'] + ngpus_per_node - 1) / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    
    # Transforms 
    train_transform = get_train_transformations()
    print(train_transform)
    train_dataset = DatasetKeyQuery(get_train_dataset(p, transform = None), train_transform, 
                                downsample_sal=not p['model_kwargs']['upsample'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=p['train_batch_size'], shuffle=(train_sampler is None),
                    num_workers=p['num_workers'], pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=collate_custom)
    print(colored('Train samples %d' %(len(train_dataset)), 'yellow'))
    print(colored(train_dataset, 'yellow'))

    # Resume from checkpoint
    if os.path.exists(p['checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['checkpoint']), 'blue'))
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(p['checkpoint'], map_location=loc)
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        if args.nvidia_apex:
            amp.load_state_dict(checkpoint['amp'])
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()

    # Main loop
    print(colored('Starting main loop', 'blue'))
    
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*10, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train 
        print('Train ...')
        eval_train = train(p, train_dataloader, model, 
                                    optimizer, epoch, amp)

        # Checkpoint
        if args.rank % ngpus_per_node == 0:
            print('Checkpoint ...')
            if args.nvidia_apex:
                torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                            'amp': amp.state_dict(), 'epoch': epoch + 1}, 
                            p['checkpoint'])

            else:
                torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                            'epoch': epoch + 1}, 
                            p['checkpoint'])


if __name__ == "__main__":
    main()
