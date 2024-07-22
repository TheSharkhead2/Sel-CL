import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms
import argparse
import os
import time
from datetime import datetime
from dataset.webvision_dataset import get_dataset

import torch.utils.data as data
from torch.utils.data import Sampler, WeightedRandomSampler
from torch import optim
from torchvision import datasets, transforms, models

import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import random
import sys

from webvision_model import build_model

sys.path.append('../utils')
from utils_noise_webvision import train_sel, train_uns, train_sup, pair_selection
from test_eval import test_eval
from queue_with_pro import queue_with_pro
from kNN_test import kNN
from MemoryMoCo import MemoryMoCo
from other_utils import save_model, TwoCropTransform, TwoTransform, tofloat, set_bn_train, moment_update
import models_webvision as mod
from lr_scheduler import get_scheduler

import wandb

from deepspeed.profiling.flops_profiler import FlopsProfiler


class DistributedWeightedSampler(Sampler):
    """
    FROM: https://github.com/pytorch/pytorch/issues/77154
    (with some modifications)

    A class for distributed data sampling with weights.

    .. note::

        For this to work correctly, global seed must be set to be the same
        across all devices.

    :param weights: A list of weights to sample with.
    :type weights: list
    :param num_samples: Number of samples in the dataset.
    :type num_samples: int
    :param replacement: Do we sample with or without replacement.
    :type replacement: bool
    :param num_replicas: Number of processes running training.
    :type num_replicas: int
    :param rank: Current device number.
    :type rank: int
    """

    def __init__(
        self,
        weights: list,
        num_samples: int = None,
        replacement: bool = True,
        num_replicas: int = None,
        rank: int = 0
    ):
        if num_replicas is None:
            num_replicas = torch.cuda.device_count()

        self.num_replicas = num_replicas
        self.num_samples_per_replica = int(
            math.ceil(len(weights) * 1.0 / self.num_replicas)
        )
        self.total_num_samples = (self.num_samples_per_replica *
                                  self.num_replicas)
        self.weights = weights
        self.replacement = replacement

        self.rank = rank

    def __iter__(self):
        """
        Produces mini sample list for current rank.

        :returns: A generator of samples.
        :rtype: Generator
        """

        if self.rank >= self.num_replicas or self.rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in "
                "the interval [0, {}]".format(self.rank, self.num_replicas - 1)
            )

        weights = self.weights.copy()
        # add extra samples to make it evenly divisible
        weights += weights[: (self.total_num_samples) - len(weights)]
        if not len(weights) == self.total_num_samples:
            raise RuntimeError(
                "There is a distributed sampler error. Num weights: {}, total size: {}".format(
                    len(weights), self.total_size
                )
            )

        # subsample for this rank
        weights = weights[self.rank:self.total_num_samples:self.num_replicas]
        weights_used = [0] * self.total_num_samples
        weights_used[self.rank:self.total_num_samples:self.num_replicas] = weights

        return iter(
            torch.multinomial(
                input=torch.as_tensor(weights_used, dtype=torch.double),
                num_samples=self.num_samples_per_replica,
                replacement=self.replacement,
            ).tolist()
        )

    def __len__(self):
        return self.num_samples_per_replica


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')

    parser.add_argument(
        '--rank',
        default=-1,
        type=int,
        help='node rank for distributed training'
    )
    parser.add_argument(
        '--dist-eval',
        action='store_true',
        default=False,
        help=('Enabling distributed evaluation'
              ' (recommended during training for faster monitor')
    )
    parser.add_argument(
        '--dist-url',
        default='env://',
        type=str,
        help='url used to set up distributed training'
    )
    parser.add_argument(
        '--dist-backend', default='nccl', type=str, help='distributed backend'
    )

    parser.add_argument("--wandb-project", default="NaN", type=str)
    parser.add_argument("--root", type=str)
    parser.add_argument("--flops_profiling", action="store_true")
    parser.add_argument("--flops_profiler_index", type=int, default=5)

    parser.add_argument("--load_from_config", action="store_true")
    parser.add_argument("--exp_path", type=str)
    parser.add_argument("--res_path", type=str)

    parser.add_argument(
        '--epoch', type=int, default=130, help='training epoches')
    parser.add_argument(
        '--warmup_way', type=str, default="sup", help='uns, sup')
    parser.add_argument(
        '--warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument(
        '--lr', '--base-learning-rate', '--base-lr', type=float, default=0.1,
        help='learning rate')
    parser.add_argument(
        '--lr-scheduler', type=str, default='cosine',
        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument(
        '--lr-warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument(
        '--lr-warmup-multiplier', type=int, default=100,
        help='warmup multiplier')
    parser.add_argument(
        '--lr-decay-epochs', type=int, default=[80, 105], nargs='+',
        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument(
        '--lr-decay-rate', type=float, default=0.1,
        help='for step scheduler. decay rate for learning rate')
    parser.add_argument(
        '--initial_epoch', type=int, default=1,
        help="Star training at initial_epoch")

    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='#images in each mini-batch')
    parser.add_argument(
        '--test_batch_size', type=int, default=100,
        help='#images in each mini-batch')
    parser.add_argument(
        '--num_classes', type=int, default=50,
        help='Number of in-distribution classes')
    parser.add_argument(
        '--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument(
        '--momentum', default=0.9, type=float, help='momentum')

    parser.add_argument(
        '--dataset', type=str, default='inat100k', help='helpful help',
        choices=[
            "inat100k"
        ]
    )

    # DIRECTORIES
    parser.add_argument(
        '--trainval_root', default='./dataset/webvision-50/',
        help='root for trainval data')
    parser.add_argument(
        '--val_root', default='./dataset/imagenet/',
        help='root for imagenet val data')
    parser.add_argument(
        '--out', type=str, default='./out/', help='Directory of the output')

    parser.add_argument(
        '--experiment_name', type=str, default='inat',
        help='name of the experiment (for the output files)')

    parser.add_argument(
        '--network', type=str, default='RN18', help='Network architecture')
    parser.add_argument(
        '--headType', type=str, default="Linear", help='Linear, NonLinear')
    parser.add_argument(
        '--low_dim', type=int, default=128,
        help='Size of contrastive learning embedding')
    parser.add_argument(
        '--seed_initialization', type=int, default=1,
        help='random seed (default: 1)')
    parser.add_argument(
        '--seed_dataset', type=int, default=42,
        help='random seed (default: 1)')

    parser.add_argument(
        '--alpha_m', type=float, default=1.0,
        help='Beta distribution parameter for mixup')
    parser.add_argument(
        '--alpha_moving', type=float, default=0.999,
        help='exponential moving average weight')
    parser.add_argument(
        '--alpha', type=float, default=0.5, help='example selection th')
    parser.add_argument(
        '--beta', type=float, default=0.5, help='pair selection th')
    parser.add_argument(
        '--uns_queue_k', type=int, default=10000,
        help='uns-cl num negative sampler')
    parser.add_argument(
        '--uns_t', type=float, default=0.1, help='uns-cl temperature')
    parser.add_argument(
        '--sup_t', default=0.1, type=float, help='sup-cl temperature')
    parser.add_argument(
        '--sup_queue_use', type=int, default=1, help='1: Use queue for sup-cl')
    parser.add_argument(
        '--sup_queue_begin', type=int, default=3,
        help='Epoch to begin using queue for sup-cl')
    parser.add_argument(
        '--queue_per_class', type=int, default=100,
        help=('Num of samples per class to store in the queue.'
              'queue size = queue_per_class*num_classes*2')
    )
    parser.add_argument(
        '--aprox', type=int, default=1,
        help=('Approximation for numerical stability taken'
              'from supervised contrastive learning')
    )
    parser.add_argument(
        '--lambda_s', type=float, default=0.01,
        help='weight for similarity loss')
    parser.add_argument(
        '--lambda_c', type=float, default=1,
        help='weight for classification loss')
    parser.add_argument(
        '--k_val', type=int, default=250, help='k for k-nn correction')

    args = parser.parse_args()
    return args


def data_config(args, transform_train, transform_test):

    trainset, testset, imagenet_set = get_dataset(
        args,
        TwoCropTransform(transform_train),
        transform_test
    )

    if args.distributed:
        train_sampler = torch.utils.data.DistributedSampler(trainset)
        test_sampler = torch.utils.data.DistributedSampler(testset)
        imagenet_sampler = torch.utils.data.DistributedSampler(imagenet_set)
    else:
        train_sampler = torch.utils.data.SequentialSampler(trainset)
        test_sampler = torch.utils.data.SequentialSampler(testset)
        imagenet_sampler = torch.utils.data.SequentialSampler(imagenet_set)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=test_sampler
    )
    imagenet_test_loader = torch.utils.data.DataLoader(
        imagenet_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=imagenet_sampler
    )
    print('############# Data loaded #############')

    return train_loader, test_loader, imagenet_test_loader, trainset


def build_models(args, device, exp_path):
    model = build_model(args, device)
    model_ema = build_model(args, device)

    print(
        'Total params: {:.2f} M'.format(
            (
                sum(
                    p.numel() for p in model.parameters()
                ) / 1000000.0
            )
        )
    )

    # copy weights from `model' to `model_ema'
    moment_update(model, model_ema, 0)
    # model_ema = None
    return model, model_ema


def main(args):
    init_distributed_mode(args)

    # add date to out path
    args.out = os.path.join(
        args.out,
        datetime.now().strftime("%d%m%y-%H.%M.%S")
    )

    if args.load_from_config:
        exp_path = args.exp_path
        res_path = args.res_path
    else:
        exp_path = os.path.join(
            args.out,
            'noise_models_' + args.network + '_{0}_SI{1}_SD{2}'.format(
                args.experiment_name,
                args.seed_initialization,
                args.seed_dataset
            )
        )
        res_path = os.path.join(
            args.out,
            'metrics' + args.network + '_{0}_SI{1}_SD{2}'.format(
                args.experiment_name,
                args.seed_initialization,
                args.seed_dataset
            )
        )

    if not os.path.isdir(res_path):
        os.makedirs(res_path)

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)

    print(args)

    # best_ac only record the best top1_ac for validation set.
    args.best_acc = 0
    best_acc5 = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fix the GPU to deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed_initialization)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed_initialization)  # GPU seed

    # python seed for image transformation
    random.seed(args.seed_initialization)

    if args.dataset == "inat100k":
        # 123.3945866481466, 126.3885961963497, 107.48126061777886
        mean = [0.4839, 0.4956, 0.4215]
        # 55.34541875861789, 54.35099612029171, 62.73211140831446
        std = [0.2170, 0.2131, 0.2460]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.2, 1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # data loader
    num_classes = args.num_classes

    train_loader, test_loader, imagenet_test_loader, trainset = data_config(
        args, transform_train, transform_test)

    model, model_ema = build_models(args, device, exp_path)

    if args.distributed:
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        model_ema.to(device)
        model_ema = torch.nn.parallel.DistributedDataParallel(
                model_ema, device_ids=[args.gpu]
            )

    if args.load_from_config:
        # load models
        load_model = torch.load(
            os.path.join(
                exp_path,
                "Sel-CL_model_" + str(args.initial_epoch) + "epoch.pth"
            )
        )
        load_model_ema = torch.load(
            os.path.join(
                exp_path,
                "Sel-CL_model_ema_" + str(args.initial_epoch) + "epoch.pth"
            )
        )

        model.load_state_dict(load_model, strict=False)
        model_ema.load_state_dict(load_model_ema, strict=False)

    uns_contrast = MemoryMoCo(
        args.low_dim, args.uns_queue_k, args.uns_t, thresh=0).cuda()

    if args.load_from_config:
        # load uns_contrast
        uns_contrast = torch.load(
            os.path.join(
                exp_path,
                "uns_contrast_" + str(args.initial_epoch) + "epoch.pth"
            )
        )

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd
    )
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    if args.load_from_config:
        # load scheduler
        load_scheduler = torch.load(
            os.path.join(
                exp_path,
                "scheduler_" + str(args.initial_epoch) + "epoch.pth"
            )
        )
        try:
            state_dic_scheduler = {
                k.replace('module.', ''): v for k, v in load_scheduler['model'].items()
            }
        except:
            state_dic_scheduler = {
                k.replace('module.', ''): v for k, v in load_scheduler.items()
            }

        scheduler.load_state_dict(state_dic_scheduler)

    if args.sup_queue_use == 1:
        queue = queue_with_pro(args, device)
    else:
        queue = []

    if args.load_from_config:
        # load queue
        queue = torch.load(
            os.path.join(
                exp_path,
                "queue_" + str(args.initial_epoch) + "epoch.pth"
            )
        )

    if args.load_from_config and args.initial_epoch >= args.warmup_epoch:
        # load selected examples
        selected_examples = np.load(
            os.path.join(res_path, "selected_examples_train.npy")
        )

    flops_profiler = FlopsProfiler(model)

    for epoch in range(args.initial_epoch, args.epoch + 1):
        st = time.time()
        print("=================>    ", args.experiment_name)

        if (epoch <= args.warmup_epoch):
            if (args.warmup_way == 'uns'):
                train_uns(
                    args,
                    scheduler,
                    model,
                    model_ema,
                    uns_contrast,
                    queue,
                    device,
                    train_loader,
                    optimizer,
                    epoch,
                    flops_profiler
                )
            else:
                if args.distributed:
                    train_selected_sampler = DistributedWeightedSampler(
                        torch.ones(len(trainset)),
                        len(trainset),
                        rank=get_rank()
                    )
                else:
                    train_selected_sampler = \
                        torch.utils.data.WeightedRandomSampler(
                            torch.ones(len(trainset)), len(trainset)
                        )

                train_selected_loader = torch.utils.data.DataLoader(
                    trainset,
                    batch_size=args.batch_size,
                    num_workers=4,
                    pin_memory=True,
                    sampler=train_selected_sampler
                )
                trainNoisyLabels = torch.LongTensor(
                    train_loader.dataset.targets).unsqueeze(1).to(device)

                train_sup(
                    args,
                    scheduler,
                    model,
                    model_ema,
                    uns_contrast,
                    queue,
                    device,
                    train_loader,
                    train_selected_loader,
                    optimizer,
                    epoch,
                    torch.eq(trainNoisyLabels, trainNoisyLabels.t()),
                    flops_profiler
                )
        else:
            if args.distributed:
                train_selected_sampler = DistributedWeightedSampler(
                    selected_examples,
                    len(selected_examples),
                    rank=get_rank()
                )
            else:
                train_selected_sampler = \
                    torch.utils.data.WeightedRandomSampler(
                       selected_examples, len(selected_examples)
                    )
            train_selected_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=args.batch_size,
                num_workers=4,
                pin_memory=True,
                sampler=train_selected_sampler
            )
            train_sel(
                args,
                scheduler,
                model,
                model_ema,
                uns_contrast,
                queue,
                device,
                train_loader,
                train_selected_loader,
                optimizer,
                epoch,
                selected_pairs,
                flops_profiler
            )

        if (epoch % 10 == 0) or (epoch == args.epoch):
            acc, acc5 = kNN(
                args,
                epoch,
                model,
                None,
                train_loader,
                test_loader,
                200,
                0.1,
                True
            )
            if acc >= args.best_acc:
                args.best_acc = acc
                best_acc5 = acc5
            print(
                ('KNN top-1 precion: {:.4f} {:.4f},'
                 'best is: {:.4f} {:.4f}').format(
                    acc*100.,
                    acc5*100.,
                    args.best_acc*100.,
                    best_acc5*100
                )
            )

        if (epoch >= args.warmup_epoch):
            print('######## Pair-wise selection ########')
            selected_examples, selected_pairs = pair_selection(
                args,
                model,
                device,
                train_loader,
                test_loader,
                epoch
            )

            selected_examples, selected_pairs = (
                selected_examples.to(device),
                selected_pairs.to(device)
            )

        _, _, val_top1, val_top5 = test_eval(
            args, model, device, test_loader)
        _, _, test_top1, test_top5 = test_eval(
            args, model, device, imagenet_test_loader)

        wandb.log({
            "val_top1_acc": val_top1,
            "val_top5_acc": val_top5,
            "test_top1_acc": test_top1,
            "test_top5_acc": test_top5
        })

        print('Epoch time: {:.2f} seconds\n'.format(time.time()-st))
        st = time.time()

        if (epoch % 10 == 0):
            snapLast = "Sel-CL_model"
            torch.save(
                model.state_dict(),
                os.path.join(exp_path, snapLast + '_'+str(epoch)+'epoch.pth')
            )
            torch.save(
                model_ema.state_dict(),
                os.path.join(
                    exp_path, snapLast + '_ema_'+str(epoch)+'epoch.pth'
                )
            )
            torch.save(
                scheduler.state_dict(),
                os.path.join(exp_path, 'scheduler_'+str(epoch)+'epoch.pth')
            )
            torch.save(
                uns_contrast,
                os.path.join(exp_path, 'uns_contrast_'+str(epoch)+'epoch.pth')
            )
            torch.save(
                queue,
                os.path.join(exp_path, 'queue_'+str(epoch)+'epoch.pth')
            )

        if (epoch == args.epoch):
            torch.save(
                model.state_dict(),
                os.path.join(exp_path, snapLast+'.pth')
            )

        if (
            ((epoch % 10 == 0) or (epoch == args.epoch))
            and epoch >= args.warmup_epoch
        ):
            np.save(
                res_path + '/' + 'selected_examples_train.npy',
                selected_examples.data.cpu().numpy()
            )


if __name__ == "__main__":
    args = parse_args()

    wandb.init(
        project=args.wandb_project,
        config={
            "epochs": args.epoch,
            "warmup_epochs": args.warmup_epoch,
            "warmup_way": args.warmup_way,
            "base_lr": args.lr,
            "lr_scheduler": args.lr_scheduler,
            "lr_warmup_epoch": args.lr_warmup_epoch,
            "lr_warmup_multiplier": args.lr_warmup_multiplier,
            "lr_decay_epochs": args.lr_decay_epochs,
            "lr_decay_rate": args.lr_decay_rate,
            "initial_epoch": args.initial_epoch,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "weight_decay": args.wd,
            "momentum": args.momentum,
            "dataset": args.dataset,
            "experiment_name": args.experiment_name,
            "low_dim": args.low_dim,
            "seed_initialization": args.seed_initialization,
            "seed_dataset": args.seed_dataset,
            "alpha_m": args.alpha_m,
            "alpha_moving": args.alpha_moving,
            "alpha": args.alpha,
            "beta": args.beta,
            "uns_queue_k": args.uns_queue_k,
            "uns_t": args.uns_t,
            "sup_t": args.sup_t,
            "sup_queue_use": args.sup_queue_use,
            "sup_queue_begin": args.sup_queue_begin,
            "queue_per_class": args.queue_per_class,
            "aprox": args.aprox,
            "lambda_s": args.lambda_s,
            "lambda_c": args.lambda_c,
            "k_val": args.k_val
        }
    )

    main(args)
