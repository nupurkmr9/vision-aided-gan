# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import scipy.linalg
import torch
import copy
import torch.distributions as tdist

from cleanfid import fid
from cleanfid.features import *
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
        else:
            score = fid.compute_fid(gen=gen, dataset_name=opts.name, dataset_res=G.img_resolution,  mode="clean", dataset_split=opts.split, batch_size=8)
        return float(score)
    
    else:
        # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
        detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
        detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

        mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real, batch_size=8).get_mean_cov()

        mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen, batch_size=8).get_mean_cov()

        if opts.rank != 0:
            return float('nan')

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        score = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(score)

#----------------------------------------------------------------------------



def sort_likelihood(opts, max_real, num_gen):
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    gen = lambda z: (G(z, None)* 127.5 + 128).clamp(0, 255).to(torch.uint8)
    
    model = build_feature_extractor('clean', device=opts.device)

    feats, latents = fid.get_model_features(gen, model, mode='clean', z_dim=512, num_gen=5000,
                            batch_size=8, device=opts.device, return_z = True)
    
    mu, sigma = get_reference_statistics(opts.name, G.img_resolution, split=opts.split)

    
    eigvals = np.linalg.eigvals(sigma)
    if not np.all(eigvals > 1e-5):
        sigma += np.eye(sigma.shape[0]) * 0.001

    distribution = tdist.MultivariateNormal(torch.from_numpy(mu).float().reshape(1,-1).to(opts.device), 
                                            torch.from_numpy(sigma).float().to(opts.device))
    distances = distribution.log_prob(torch.from_numpy(feats).to(opts.device))
    indices = np.argsort(distances.cpu().numpy())
    latents = torch.stack([latents[indices[i]] for i in range(30)],0)
    images = torch.clamp( G( latents, None)*0.5+0.5 , 0., 1.)
    return images

#----------------------------------------------------------------------------