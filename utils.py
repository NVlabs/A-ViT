# --------------------------------------------------------
# Copyright (C) 2022 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2022 paper
# A-ViT: Adaptive Tokens for Efficient Vision Transformer
# Hongxu Yin, Arash Vahdat, Jose M. Alvarez, Arun Mallya, Jan Kautz,
# and Pavlo Molchanov
# --------------------------------------------------------


"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

import numpy as np

from torch import nn


# The following snippet is taken from:
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

# The following snippet is taken from:
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

# The following snippet is taken from:
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)

# The following snippet is taken from:
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
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

# The following snippet is taken from:
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

# The following snippet is taken from:
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

# The following snippet is taken from:
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

# The following snippet is taken from:
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
def is_main_process():
    return get_rank() == 0

# The following snippet is taken from:
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

# The following snippet is taken from:
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
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



def get_distribution_target(mode='gaussian', length=12, max=1, standardized=True, target_depth=8, buffer=0.02):
    """
    This generates the target distributional prior
    """
    # this gets the distributional target to regularize the ACT halting scores towards
    if mode == 'gaussian':
        from scipy.stats import norm
        # now get a serios of length
        data = np.arange(length)
        data = norm.pdf(data, loc=target_depth, scale=1)

        if standardized:
            print('\nReshaping distribution to be top-1 sum 1 - error at {}'.format(buffer))
            scaling_factor = (1.-buffer) / sum(data[:target_depth])
            data *= scaling_factor

        return data

    elif mode == 'lognorm':
        from scipy.stats import lognorm

        data = np.arange(length)
        data = lognorm.pdf(data, s=0.99)

        if standardized:
            print('\nReshaping distribution to be top-1 sum 1 - error at {}'.format(buffer))
            scaling_factor = (1.-buffer) / sum(data[:target_depth])
            data *= scaling_factor

        print('\nForming distribution at:', data)
        return data

    elif mode == 'skewnorm':
        from scipy.stats import skewnorm
        # now get a serios of length
        data = np.arange(1,length)
        data = skewnorm.pdf(data, a=-4, loc=target_depth)
        return data

    else:
        print('Get distributional prior not implemented!')
        raise NotImplementedError

# The following snippet is taken from:
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
class RegularizationLoss():
    """
    ## Regularization loss
    $$L_{Reg} = \mathop{KL} \Big(p_n \Vert p_G(\lambda_p) \Big)$$
    $\mathop{KL}$ is the [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).
    $p_G$ is the [Geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution) parameterized by
    $\lambda_p$. *$\lambda_p$ has nothing to do with $\lambda_n$; we are just sticking to same notation as the paper*.
    $$Pr_{p_G(\lambda_p)}(X = k) = (1 - \lambda_p)^k \lambda_p$$.
    The regularization loss biases the network towards taking $\frac{1}{\lambda_p}$ steps and incentivies non-zero probabilities
    for all steps; i.e. promotes exploration.
    """

    def __init__(self, lambda_p: float, max_steps: int = 12, args=None):
        """
        * `lambda_p` is $\lambda_p$ - the success probability of geometric distribution
        * `max_steps` is the highest $N$; we use this to pre-compute $p_G(\lambda_p)$
        """
        super().__init__()

        # Empty vector to calculate $p_G(\lambda_p)$
        p_g = torch.zeros((max_steps,))
        # $(1 - \lambda_p)^k$
        not_halted = 1.
        # Iterate upto `max_steps`
        for k in range(max_steps):
            # $$Pr_{p_G(\lambda_p)}(X = k) = (1 - \lambda_p)^k \lambda_p$$
            p_g[k] = not_halted * lambda_p
            # Update $(1 - \lambda_p)^k$
            not_halted = not_halted * (1 - lambda_p)

        # Save $Pr_{p_G(\lambda_p)}$
        self.p_g = nn.Parameter(p_g, requires_grad=False).cuda()
        self.p_g = self.p_g.expand(args.batch_size, max_steps).permute(1,0)

        # KL-divergence loss
        self.kl_div = nn.KLDivLoss(reduction='batchmean').cuda()


    def forward(self, p):
        """
        * `p` is $p_1 \dots p_N$ in a tensor of shape `[N, batch_size]`
        """
        p = torch.clamp(torch.stack(p), 0.01, 0.99)

        return self.kl_div(p.log(), self.p_g)


def h_to_p(h_lst):
    p_lst = []
    accum = 1
    for i in range(len(h_lst)):
        p_lst.append(h_lst[i] * accum)
        accum = accum * (1.-h_lst[i])
    return p_lst


### Testing Only
if __name__ == '__main__':
    get_distribution_target()
