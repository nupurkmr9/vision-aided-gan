# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# modified for "Ensembling Off-the-shelf Models for GAN Training"

import os
import click
import re
import json
import tempfile
import torch
import dnnlib
import argparse
import importlib

from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------


class UserError(Exception):
    pass

#----------------------------------------------------------------------------


def setup_training_loop_kwargs(
    # General options (not included in desc).
    gpus       = None, # Number of GPUs: <int>, default = 1 gpu
    snap       = None, # Snapshot interval: <int>, default = 50 ticks
    metrics    = None, # List of metric names: [], ['fid50k_full'] (default), ...
    seed       = None, # Random seed: <int>, default = 0

    # Dataset.
    data       = None, # Training dataset (required): <path>
    cond       = None, # Train conditional model based on dataset labels: <bool>, default = False
    subset     = None, # Train with only N images: <int>, default = all
    mirror     = None, # Augment dataset with x-flips: <bool>, default = False

    # Base config.
    cfg        = None, # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
    gamma      = None, # Override R1 gamma: <float>
    pl_weight  = None, # Override R1 gamma: <float>
    g_reg_interval  = None, # Override R1 gamma: <float>
    d_reg_interval  = None, # Override R1 gamma: <float>
    kimg       = None, # Override training duration: <int>
    batch      = None, # Override batch size: <int>
    batch_gpu  = None, # Override batch size per gpu: <int>

    # Discriminator augmentation.
    diffaugment= None, # Comma-separated list of DiffAugment policy, default = 'color,translation,cutout'
    aug        = None, # Augmentation mode: 'ada' (default), 'noaug', 'fixed'
    p          = None, # Specify p for 'fixed' (required): <float>
    target     = None, # Override ADA target for 'ada': <float>, default = depends on aug
    augpipe    = None, # Augmentation pipeline: 'blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc' (default), ..., 'bgcfnc'

    # Transfer learning.
    resume     = None, # Load previous network: 'noresume' (default), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>
    freezed    = None, # Freeze-D: <int>, default = 0 discriminator layers

    # Performance options (not included in desc).
    fp32       = None, # Disable mixed-precision training: <bool>, default = False
    nhwc       = None, # Use NHWC memory format with FP16: <bool>, default = False
    allow_tf32 = None, # Allow PyTorch to use TF32 for matmul and convolutions: <bool>, default = False
    nobench    = None, # Disable cuDNN benchmarking: <bool>, default = False
    workers    = None, # Override number of DataLoader workers: <int>, default = 3
    
    # Visison-aided-gan training configs.
    cv         = None, # cv supervision on images
    warmup     = False, # warmup if training from scratch
    cv_loss    = None, # cv loss
    augcv      = None, # Augmentation mode for CV input: 'ada' (default), 'noaug', 'fixed'
    augpipecv  = None, # Augmentation pipeline for CV: 
    ada_target_cv = None, #target augmentation p for CV model

    # miscellaneous 
    wandb_log     = None, # use wandb logging: setup entity and project name in training/trainin_loop.py
    exact_resume  = 0,    # if 1 resume augmentation pipeline and optimizer 
    clean         = None, # FID clean or not
    **kwargs,

):
    
    args = dnnlib.EasyDict()

    # ------------------------------------------
    # General options: gpus, snap, metrics, seed
    # ------------------------------------------
    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    if snap is None:
        snap = 25
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    if metrics is None:
        metrics = ['fid50k_full']
    assert isinstance(metrics, list)
    if not all(metric_main.is_valid_metric(metric) for metric in metrics):
        raise UserError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    args.metrics = metrics

    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    # -----------------------------------
    # Dataset: data, cond, subset, mirror
    # -----------------------------------

    assert data is not None
    assert isinstance(data, str)
    args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # subclass of training.dataset.Dataset
        args.training_set_kwargs.resolution = training_set.resolution # be explicit about resolution
        args.training_set_kwargs.use_labels = training_set.has_labels # be explicit about labels
        args.training_set_kwargs.max_size = len(training_set) # be explicit about dataset size
        desc = training_set.name
        del training_set # conserve memory
       

    except IOError as err:
        raise UserError(f'--data: {err}')

    if cond is None:
        cond = False
    assert isinstance(cond, bool)
    if cond:
        if not args.training_set_kwargs.use_labels:
            raise UserError('--cond=True requires labels specified in dataset.json')
        desc += '-cond'
    else:
        args.training_set_kwargs.use_labels = False

    if subset is not None:
        assert isinstance(subset, int)
        if not 1 <= subset <= args.training_set_kwargs.max_size:
            raise UserError(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
        desc += f'-subset{subset}'
        if subset < args.training_set_kwargs.max_size:
            args.training_set_kwargs.max_size = subset
            args.training_set_kwargs.random_seed = args.random_seed

    if mirror is None:
        mirror = False
    assert isinstance(mirror, bool)
    if mirror:
        desc += '-mirror'
        args.training_set_kwargs.xflip = True

    # ------------------------------------
    # Base config: cfg, gamma, kimg, batch
    # ------------------------------------

    if cfg is None:
        cfg = 'auto'
    assert isinstance(cfg, str)
    desc += f'-{cfg}'

    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
        'stylegan2': dict(ref_gpus=4,  kimg=25000,  mb=16, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
        'paper256':  dict(ref_gpus=2,  kimg=25000,  mb=16, mbstd=4,  fmaps=0.5, lrate=0.002, gamma=1,    ema=10,  ramp=None, map=8),
        'paper512':  dict(ref_gpus=4,  kimg=25000,  mb=16, mbstd=4,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper1024': dict(ref_gpus=4,  kimg=25000,  mb=16, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
        'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2), 
        'paper256_2fmap':  dict(ref_gpus=2,  kimg=25000,  mb=16, mbstd=4,  fmaps=1, lrate=0.002, gamma=1,    ema=10,  ramp=None, map=8),
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        desc += f'{gpus:d}'
        spec.ref_gpus = gpus
        res = args.training_set_kwargs.resolution
        spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
        spec.mbstd = min(spec.mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32

    args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    
    args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    
    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
    args.G_kwargs.mapping_kwargs.num_layers = spec.map
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4 # enable mixed-precision training
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd
    
    
    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', r1_gamma=spec.gamma)

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp

    if cfg == 'cifar':
        args.loss_kwargs.pl_weight = 0 # disable path length regularization
        args.loss_kwargs.style_mixing_prob = 0 # disable style mixing
        args.D_kwargs.architecture = 'orig' # disable residual skip connections
    if pl_weight is not None:
        args.loss_kwargs.pl_weight = pl_weight
    if g_reg_interval == 0:
        args.G_reg_interval = None
    if d_reg_interval ==0:
        args.D_reg_interval = None
    
    if gamma is not None:
        assert isinstance(gamma, float)
        if not gamma >= 0:
            raise UserError('--gamma must be non-negative')
        desc += f'-gamma{gamma:g}'
        args.loss_kwargs.r1_gamma = gamma

    if kimg is not None:
        assert isinstance(kimg, int)
        if not kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{kimg:d}'
        args.total_kimg = kimg

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{batch}'
        args.batch_size = batch
        args.batch_gpu = batch_gpu or  batch // gpus

    # ---------------------------------------------------
    # Discriminator augmentation: aug, p, target, augpipe
    # ---------------------------------------------------
    
    if diffaugment:
        args.loss_kwargs.diffaugment = diffaugment
        aug = 'noaug'
        desc += '-{}'.format(diffaugment.replace(',', '-'))
        
    if aug is None:
        aug = 'ada'
    else:
        assert isinstance(aug, str)
        desc += f'-{aug}'

    if aug == 'ada':
        args.ada_target = 0.6
    elif aug == 'noaug':
        pass
    elif aug == 'fixed':
        if p is None:
            raise UserError(f'--aug={aug} requires specifying --p')
    else:
        raise UserError(f'--aug={aug} not supported')
        
    if p is not None:
        assert isinstance(p, float)
        if aug != 'fixed':
            raise UserError('--p can only be specified with --aug=fixed')
        if not 0 <= p <= 1:
            raise UserError('--p must be between 0 and 1')
        desc += f'-p{p:g}'
        args.augment_p = p

    if target is not None:
        assert isinstance(target, float)
        if aug != 'ada':
            raise UserError('--target can only be specified with --aug=ada')
        if not 0 <= target <= 1:
            raise UserError('--target must be between 0 and 1')
        desc += f'-target{target:g}'
        args.ada_target = target

    assert augpipe is None or isinstance(augpipe, str)
    if augpipe is None:
        augpipe = 'bgc'
    else:
        if aug == 'noaug':
            raise UserError('--augpipe cannot be specified with --aug=noaug')
        desc += f'-{augpipe}'


    augpipe_specs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }
    
    assert augpipe in augpipe_specs
    if aug != 'noaug':
        args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[augpipe])

    # ----------------------------------
    # Vision-aided-Adversarial loss
    # ----------------------------------

    args.cvD_kwargs = None
    args.warmup = warmup
    if cv is not None:
        desc += '-{}'.format(cv)
        desc += '-cv_loss_{}'.format(cv_loss)

        output = cv.split('output-')[1]
        cv = cv.split('-output')[0].split('input-')[1]

        args.loss_kwargs.cv_loss = cv_loss
        args.loss_kwargs.num_cv_models = len(cv.split('+'))

        args.cvD_kwargs = dnnlib.EasyDict(class_name='vision_module.cv_discriminator.Discriminator', cv_type= cv, output_type=output)
        args.cvD_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
        
        # ----------------------------------
        # Pretrained model based Discriminator augmentation: augcv, p, target, augpipecv
        # ----------------------------------

        if augcv is not None:
            if 'noaug' in augcv:
                args.augment_kwargs_cv = None
            elif 'diffaug' in augcv:
                args.loss_kwargs.augment_pipe_cv = augcv
                args.augment_kwargs_cv = None
                desc += '-augcv_diffaug'
            elif augcv =='ada':
                args.ada_target_cv = ada_target_cv
                desc += f'-augcv_{augcv}'
                if augpipecv is None:
                    augpipecv = 'bgc'
                args.augment_kwargs_cv = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[augpipecv])
    
    # ----------------------------------
    # Transfer learning: resume, freezed
    # ----------------------------------

    resume_specs = {
        'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
        'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
        'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
        'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
        'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
    }

    assert resume is None or isinstance(resume, str)
    if resume is None:
        resume = 'noresume'
    elif resume == 'noresume':
        desc += '-noresume'
    elif resume in resume_specs:
        desc += f'-resume{resume}'
        args.resume_pkl = resume_specs[resume] # predefined url
    else:
        desc += '-resumecustom'
        args.resume_pkl = resume # custom path or url

    if resume != 'noresume':
        if cv is None:
            args.ada_kimg = 100 # make ADA react faster at the beginning. We remove this when we add our loss on a model trained with standard adversarial loss till some iterations 
        args.ema_rampup = None # disable EMA rampup
        args.exact_resume = exact_resume
        
    
    if freezed is not None:
        assert isinstance(freezed, int)
        if not freezed >= 0:
            raise UserError('--freezed must be non-negative')
        desc += f'-freezed{freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = freezed

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------

    if fp32 is None:
        fp32 = False
    assert isinstance(fp32, bool)
    if fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

    if nhwc is None:
        nhwc = False
    assert isinstance(nhwc, bool)
    if nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

    if nobench is None:
        nobench = False
    assert isinstance(nobench, bool)
    if nobench:
        args.cudnn_benchmark = False

    if allow_tf32 is None:
        allow_tf32 = False
    assert isinstance(allow_tf32, bool)
    if allow_tf32:
        args.allow_tf32 = True

    if workers is not None:
        assert isinstance(workers, int)
        if not workers >= 1:
            raise UserError('--workers must be at least 1')
        args.data_loader_kwargs.num_workers = workers
        
    args.wandb_log = wandb_log
    args.clean = clean
    args.desc = desc
    
    return desc, args

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    importlib.reload(training_stats)
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **args)

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser('main')

    # General options.
    parser.add_argument('--outdir', help='Where to save the results', required=True, metavar='DIR')
    parser.add_argument('--gpus', help='Number of GPUs to use [default: 1]', type=int, metavar='INT')
    parser.add_argument('--snap', help='Snapshot interval [default: 50 ticks]', type=int, metavar='INT')
    parser.add_argument('--metrics', help='Comma-separated list or "none" [default: fid50k_full]', type=CommaSeparatedList())
    parser.add_argument('--seed', help='Random seed [default: 0]', type=int, metavar='INT')
    parser.add_argument('--dry-run', help='Print training options and exit', action='store_true', default=False)

    # Dataset.
    parser.add_argument('--data', help='Training data (directory or zip)', metavar='PATH', required=True)
    parser.add_argument('--cond', help='Train conditional model based on dataset labels [default: false]', type=bool, metavar='BOOL')
    parser.add_argument('--subset', help='Train with only N images [default: all]', type=int, metavar='INT')
    parser.add_argument('--mirror', help='Enable dataset x-flips [default: false]', type=bool, metavar='BOOL')

    # Base config.
    parser.add_argument('--cfg', help='Base config [default: auto]', type=click.Choice(['auto', 'stylegan2', 'paper256', 'paper256_2fmap', 'paper512', 'paper1024', 'cifar']))
    parser.add_argument('--gamma', help='Override R1 gamma', type=float)
    parser.add_argument('--pl-weight', help='Override G path regularization', type=float)
    parser.add_argument('--g-reg-interval', help='lazy regularization G interval', type=int, metavar='INT')
    parser.add_argument('--d-reg-interval', help='lazy regularization D interval', type=int, metavar='INT')
    parser.add_argument('--kimg', help='Override training duration', type=int, metavar='INT')
    parser.add_argument('--batch', help='Override batch size', type=int, metavar='INT')
    parser.add_argument('--batch-gpu', help='Override batch size per gpu', type=int, metavar='INT')

    # Discriminator augmentation.
    parser.add_argument('--diffaugment', help='Comma-separated list of DiffAugment policy [default: color,translation,cutout]', type=str)
    parser.add_argument('--aug', help='Augmentation mode [default: ada]', type=click.Choice(['noaug', 'ada', 'fixed']))
    parser.add_argument('--p', help='Augmentation probability for --aug=fixed', type=float)
    parser.add_argument('--target', help='ADA target value for --aug=ada', type=float)
    parser.add_argument('--augpipe', help='Augmentation pipeline [default: bgc]', type=click.Choice(['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']))

    # Transfer learning.
    parser.add_argument('--resume', help='Resume training [default: noresume]', metavar='PKL')
    parser.add_argument('--freezed', help='Freeze-D [default: 0 layers]', type=int, metavar='INT')

    # Performance options.
    parser.add_argument('--fp32', help='Disable mixed-precision training', type=bool, metavar='BOOL')
    parser.add_argument('--nhwc', help='Use NHWC memory format with FP16', type=bool, metavar='BOOL')
    parser.add_argument('--nobench', help='Disable cuDNN benchmarking', type=bool, metavar='BOOL')
    parser.add_argument('--allow-tf32', help='Allow PyTorch to use TF32 internally', type=bool, metavar='BOOL')
    parser.add_argument('--workers', help='Override number of DataLoader workers', type=int, metavar='INT')

    #Vision-aided adversarial loss options
    parser.add_argument('--cv', help='CV model [default: None]', type=str) 
    parser.add_argument('--warmup', help='if training from scratch, train with standard adversarial loss for 500k images and then introduce vision-aided-loss', type=float, default=0.) 
    parser.add_argument('--cv-loss', default= '', help='CV loss', type=str) 
    parser.add_argument('--augcv', help='Augmentation mode [default: None]', type=str)
    parser.add_argument('--augpipecv', default='bgc', help='Augmentation pipeline [default: bgc]', type=str)
    parser.add_argument('--ada-target-cv', default=0.3, help='Augmentation target probability', type=float)
    parser.add_argument('--exact-resume', help='0: only resume model weights, 1: resume model weights ensuring exact match and augpipe, 2: resume model weights, optimizer and augpipe', type=int, default=0, metavar='INT')

    # miscellaneous
    parser.add_argument('--wandb-log', help='wandb logging instead of tensorboard logging', type=bool, metavar='BOOL', default=False)
    parser.add_argument('--clean', help='FID evaluation using clean-fid (https://github.com/GaParmar/clean-fid)', type=bool, metavar='BOOL', default=False)

    return parser

def main(args):
    """Train a GAN using the techniques described in the paper
    "Ensembling Off-the-shelf Models for GAN Training".

    Examples:

    \b
    # Train StyleGAN2 on custom dataset using 1 GPU.
    python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1 

    \b
    Base configs (--cfg):
      auto       Automatically select reasonable defaults based on resolution
                 and GPU count. Good starting point for new datasets.
      paper256         Reproduce results for 256 resolution dataset
      paper256_2fmap   Reproduce results for 256 resolution dataset with 2x feature channel in G.
      paper512         Reproduce results for 512 resolution dataset
      paper1024        Reproduce results for MetFaces at 1024x1024.
      stylegan2        Reproduce results for StyleGAN2 config F at 1024x1024.

    \b
    Transfer learning source networks (--resume):
      ffhq512        FFHQ trained at 512x512 resolution.
      ffhq1024       FFHQ trained at 1024x1024 resolution.
      <PATH or URL>  Custom network pickle.

     \b
    Computer vision pretrained networks (--cv):
      input-<model_name>-output-<output_type>
      model_name: clip, dino, seg_ade, det_coco, face_seg, face_normals, vgg, swin
      output_type: conv_multi_level for training (discriminator head is over the spatial features of pretrained model)
      output_type: pool for model selection (to enable linear classifier on the feature vector)
      for multiple model training: model_name and output_type is + separated, i.e. clip+dino+swin; and conv_multi_level+conv_multi_level+conv. 


     \b
    Loss function for computer vision model bsaed discriminator (--cv-loss):
      sigmoid: standard loss used for all models except CLIP and DINO
      multilevel: multi-level loss used in CLIP and DINO
      sigmoid_s: sidmoid loss with one sided label smoothing
      multilevel_s: multi level loss with one sided label smoothing
      for multiple model training: loss is + separated names, i.e. multilevel_s+sigmoid_s+sigmoid_s

     \b
    Augmentation for computer vision model bsaed discriminator (--augcv):
      can be either "ada" or "diffaug-policy" where policy is a comma separated augmentation list e.g. color,translation,cutout

     \b
    Augmentation for original discriminator (--augcv, --diffaugment):
      --augcv ada: For ADA augmentation
      --diffaugment <policy>: For DiffAugment, where example policy is 'color,translation,cutout'

    """

    dnnlib.util.Logger(should_flush=True)
    outdir = args.outdir
    dry_run = args.dry_run
    # Setup training options.
    try:
        run_desc, args = setup_training_loop_kwargs(**vars(args))
    except UserError as err:
        print(err)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    args.desc = f'{cur_run_id:05d}-{run_desc}'
    assert not os.path.exists(args.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training data:      {args.training_set_kwargs.path}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Number of GPUs:     {args.num_gpus}')
    print(f'Number of images:   {args.training_set_kwargs.max_size}')
    print(f'Image resolution:   {args.training_set_kwargs.resolution}')
    print(f'Conditional model:  {args.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:    {args.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------            

if __name__ == "__main__":
    args = get_args_parser()
    main(args.parse_args())

#----------------------------------------------------------------------------

