# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import copy
import scipy.linalg
import torch
from cleanfid import fid
from . import metric_utils
#----------------------------------------------------------------------------

def compute_fid(opts, max_real, num_gen):
    if opts.clean:        
        G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
        gen = lambda z: (G(z, None)* 127.5 + 128).clamp(0, 255).to(torch.uint8)
        if 'metfaces' in opts.dataset_kwargs.path:
            score = fid.compute_fid(gen=gen, dataset_name="metfaces", dataset_res=1024,  mode="clean", dataset_split="train", batch_size=8)
        elif 'afhqdog' in opts.dataset_kwargs.path:
            score = fid.compute_fid(gen=gen, dataset_name="afhq_dog", dataset_res=512, mode="clean", dataset_split="train", batch_size=8)
        elif 'afhqcat' in opts.dataset_kwargs.path:
            score = fid.compute_fid(gen=gen, dataset_name="afhq_cat", dataset_res=512, mode="clean", dataset_split="train", batch_size=8)
        elif 'afhqwild' in opts.dataset_kwargs.path:
            score = fid.compute_fid(gen=gen, dataset_name="afhq_wild", dataset_res=512, mode="clean", dataset_split="train", batch_size=8)
        elif 'church' in opts.dataset_kwargs.path:
            score = fid.compute_fid(gen=gen, dataset_name="lsun_church", dataset_res=256,  mode="clean", dataset_split="trainfull", batch_size=32)
        elif 'cat' in opts.dataset_kwargs.path:
            score = fid.compute_fid(gen=gen, dataset_name="lsun_cat", dataset_res=256,  mode="clean", dataset_split="trainfull", batch_size=32)
        elif 'horse' in opts.dataset_kwargs.path:
            score = fid.compute_fid(gen=gen, dataset_name="lsun_horse", dataset_res=256, mode="clean", dataset_split="trainfull", batch_size=32)
        elif 'ffhq1024' in opts.dataset_kwargs.path:
            score = fid.compute_fid(gen=gen, dataset_name="ffhq", dataset_res=1024, mode="clean", dataset_split="trainval70k", batch_size=8)
        elif 'ffhq' in opts.dataset_kwargs.path:
            score = fid.compute_fid(gen=gen, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k", batch_size=32)
        return float(score)

    else:
        # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
        detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

        mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

        mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

        if opts.rank != 0:
            return float('nan')

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        score = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(score)

#----------------------------------------------------------------------------
