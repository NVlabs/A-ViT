# --------------------------------------------------------
# Copyright (C) 2022 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2022 paper
# A-ViT: Adaptive Tokens for Efficient Vision Transformer
# Hongxu Yin, Arash Vahdat, Jose M. Alvarez, Arun Mallya, Jan Kautz,
# and Pavlo Molchanov
# --------------------------------------------------------

# The following snippet is initially based on:
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# The code is modified to accomodate A-ViT training

"""
Train and eval functions used in main_act.py for A-ViT training and eval
"""

import math
import sys
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from losses import DistillationLoss
import utils
from timm.utils.utils import *
import numpy as np
import os
from utils import RegularizationLoss
import pickle
import heapq, random
from PIL import Image
import cv2


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None, tf_writer=None):

    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # temporarily disabled for act
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            # outputs, rho, cnt = model(samples)
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

            # now get the token rhos
            rho_token = torch.mean(model.module.rho_token)
            # for analysis and keeping track purpose
            cnt_token = model.module.counter_token.data.cpu().numpy()
            # for analysis and keeping track purpose
            cnt_token_diff = (torch.max(model.module.counter_token, dim=-1)[0]-torch.min(model.module.counter_token, dim=-1)[0]).data.cpu().numpy()

        model.module.batch_cnt += 1

        # Ponder loss
        ponder_loss_token = torch.mean(rho_token) * args.ponder_token_scale
        loss += ponder_loss_token

        # Distributional prior
        if args.distr_prior_alpha > 0.:

            # KL loss
            halting_score_distr = torch.stack(model.module.halting_score_layer)
            halting_score_distr = halting_score_distr / torch.sum(halting_score_distr)
            halting_score_distr = torch.clamp(halting_score_distr, 0.01, 0.99)
            distr_prior_loss = args.distr_prior_alpha * model.module.kl_loss(halting_score_distr.log(), model.module.distr_target)

            if distr_prior_loss.item() > 0.:
                loss += distr_prior_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)


        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        if 1:
            # update logger
            metric_logger.update(cnt_token_mean=float(np.mean(cnt_token)))
            metric_logger.update(cnt_token_max=float(np.max(cnt_token)))
            metric_logger.update(cnt_token_min=float(np.min(cnt_token)))
            metric_logger.update(cnt_token_diff=float(np.mean(cnt_token_diff)))
            metric_logger.update(ponder_loss_token=ponder_loss_token.item())
            metric_logger.update(remaining_compute=float(np.mean(cnt_token/12.)))

            if args.distr_prior_alpha > 0.:
                metric_logger.update(distri_prior_loss=distr_prior_loss.item())


        if tf_writer is not None and torch.cuda.current_device()==0:
            if model.module.batch_cnt % print_freq == 0:
                tf_writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], model.module.batch_cnt)
                tf_writer.add_scalar('train/loss', loss_value, model.module.batch_cnt)
                tf_writer.add_scalar('train/cnt_token_mean', float(np.mean(cnt_token)), model.module.batch_cnt)
                tf_writer.add_scalar('train/cnt_token_max', float(np.max(cnt_token)), model.module.batch_cnt)
                tf_writer.add_scalar('train/cnt_token_min', float(np.min(cnt_token)), model.module.batch_cnt)
                tf_writer.add_scalar('train/avg_cnt_token_diff', float(np.mean(cnt_token_diff)), model.module.batch_cnt)
                tf_writer.add_scalar('train/ponder_loss_token', ponder_loss_token.item(), model.module.batch_cnt)
                tf_writer.add_scalar('train/expected_depth_ratio', float(np.mean(cnt_token/12.)), model.module.batch_cnt)
                if args.distr_prior_alpha > 0.:
                    tf_writer.add_scalar('train/distr_prior_loss', distr_prior_loss.item(), model.module.batch_cnt)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, tf_writer=None, args=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    cnt_token, cnt_token_diff = None, None

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        if cnt_token is None:
            cnt_token = model.module.counter_token.data.cpu().numpy()
        else:
            cnt_token = np.concatenate((cnt_token, model.module.counter_token.data.cpu().numpy()))

        if cnt_token_diff is None:
            cnt_token_diff = (torch.max(model.module.counter_token, dim=-1)[0]-torch.min(model.module.counter_token, dim=-1)[0]).data.cpu().numpy()
        else:
            cnt_token_diff = np.concatenate((cnt_token_diff, \
            (torch.max(model.module.counter_token, dim=-1)[0]-torch.min(model.module.counter_token, dim=-1)[0]).data.cpu().numpy()))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    if tf_writer is not None and torch.cuda.current_device()==0:
        # writing all values
        tf_writer.add_scalar('test/acc_top1', metric_logger.acc1.global_avg, epoch)
        tf_writer.add_scalar('test/acc_top5', metric_logger.acc5.global_avg, epoch)
        tf_writer.add_scalar('test/loss', metric_logger.loss.global_avg, epoch)
        tf_writer.add_scalar('test/cnt_token_mean', float(np.mean(cnt_token)), model.module.batch_cnt)
        tf_writer.add_scalar('test/cnt_token_max', float(np.max(cnt_token)), model.module.batch_cnt)
        tf_writer.add_scalar('test/cnt_token_min', float(np.min(cnt_token)), model.module.batch_cnt)
        tf_writer.add_scalar('test/avg_cnt_token_diff', float(np.mean(cnt_token_diff)), model.module.batch_cnt)
        tf_writer.add_scalar('test/expected_depth_ratio', float(np.mean(cnt_token/12)), model.module.batch_cnt)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    # snippet for merging and visualization
    h_min = max(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def merge_image(im1, im2):
    # snippet for merging and visualization
    h_margin = 54
    v_margin = 80
    im2 = im2[h_margin+5:480-h_margin, v_margin:640-v_margin]
    return hconcat_resize_min([im1, im2])



@torch.no_grad()
def visualize(data_loader, model, device, epoch, tf_writer=None, args=None):
    import torchvision.utils as vutils
    import matplotlib.pyplot as plt
    from PIL import Image

    # this snipet visualize the token depth distribution of an avit model
    # more particular, it saves the image with the largset token depth std. per imagenet class
    # in validation set.

    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Visualize:'

    # switch to evaluation mode
    model.eval()
    save_image = True

    # amid imagenet class separation for best visualization, assert batch size is 10
    # such that no validation images overlap in classes
    assert args.batch_size==50
    class_set = set()

    for images, target in metric_logger.log_every(data_loader, 100, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        cnt_token = model.module.counter_token.data.cpu().numpy()

        # this tries to save images
        if save_image:

            cnt_token_std_lst = np.std(cnt_token, axis=-1)
            value = np.max(cnt_token_std_lst)
            key = np.argmax(cnt_token_std_lst)

            # this part fetches most sensitive samples per class
            tmp_set = set(target.data.cpu().numpy())

            if not all([x in class_set for x in tmp_set]):
                print('Now visualizing token depth for class {}/1000.'.format(target[key].data.item()))
                max_std = 0

            class_set = class_set | tmp_set

            if value >= max_std:

                max_std=value
                idx=key

                file_path = "./token_act_visualization/"
                if not os.path.exists(file_path):
                    os.makedirs(file_path)

                target_token = cnt_token[idx,1:]
                array = np.reshape(target_token, (14, 14))

                plt.imshow(array, cmap='hot', interpolation='nearest')
                plt.axis('off')
                cb=plt.colorbar(shrink=0.8)

                if 1:
                    # save token depth heat map
                    plt.savefig(file_path + 'class{}_token_depth.jpg'.format(target[idx].data.item()))
                if 1:
                    # save original image
                    vutils.save_image(images[idx].data, file_path + 'class{}_ref.jpg'.format(target[idx].data.item()),
                                          normalize=True, scale_each=True)
                if 1:
                    # save concatenated image
                    # note that this snippet is not fully optimized in speed
                    im1 = cv2.imread(file_path + 'class{}_ref.jpg'.format(target[idx].data.item()))
                    im2 = cv2.imread(file_path + 'class{}_token_depth.jpg'.format(target[idx].data.item()))

                    if im1 is not None and im2 is not None:
                        cv2.imwrite(file_path + 'class{}_combined.jpg'.format(target[idx].data.item()), merge_image(im1, im2))

                cb.remove()

    print('Visualization done.')
    exit()

    return
