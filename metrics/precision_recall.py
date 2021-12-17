# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Precision/Recall (PR) from the paper "Improved Precision and Recall
Metric for Assessing Generative Models". Matches the original implementation
by Kynkaanniemi et al. at
https://github.com/kynkaat/improved-precision-and-recall-metric/blob/master/precision_recall.py"""

import torch
import lpips
import dnnlib

from . import metric_utils

#----------------------------------------------------------------------------

def compute_distances(row_features, col_features, num_gpus, rank, col_batch_size):
    assert 0 <= rank < num_gpus
    num_cols = col_features.shape[0]
    num_batches = ((num_cols - 1) // col_batch_size // num_gpus + 1) * num_gpus
    col_batches = torch.nn.functional.pad(col_features, [0, 0, 0, -num_cols % num_batches]).chunk(num_batches)
    dist_batches = []
    for col_batch in col_batches[rank :: num_gpus]:
        dist_batch = torch.cdist(row_features.unsqueeze(0), col_batch.unsqueeze(0))[0]
        for src in range(num_gpus):
            dist_broadcast = dist_batch.clone()
            if num_gpus > 1:
                torch.distributed.broadcast(dist_broadcast, src=src)
            dist_batches.append(dist_broadcast.cpu() if rank == 0 else None)
    return torch.cat(dist_batches, dim=1)[:, :num_cols] if rank == 0 else None

#----------------------------------------------------------------------------

def compute_pr_from_feats(real_features, gen_features, nhood_size=3, row_batch_size=10000, col_batch_size=10000, num_gpus=1, rank=0):
    results = dict()
    for name, manifold, probes in [('precision', real_features, gen_features), ('recall', gen_features, real_features)]:
        kth = []
        for manifold_batch in manifold.split(row_batch_size):
            dist = compute_distances(row_features=manifold_batch, col_features=manifold, num_gpus=num_gpus, rank=rank, col_batch_size=col_batch_size)
            kth.append(dist.to(torch.float32).kthvalue(nhood_size + 1).values.to(torch.float16) if rank == 0 else None)
        kth = torch.cat(kth) if rank == 0 else None
        pred = []
        for probes_batch in probes.split(row_batch_size):
            dist = compute_distances(row_features=probes_batch, col_features=manifold, num_gpus=num_gpus, rank=rank, col_batch_size=col_batch_size)
            pred.append((dist <= kth).any(dim=1) if rank == 0 else None)
        results[name] = float(torch.cat(pred).to(torch.float32).mean() if rank == 0 else 'nan')
    return results['precision'], results['recall']

#----------------------------------------------------------------------------



def compute_pr(opts, max_real, num_gen, nhood_size, row_batch_size, col_batch_size):
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    detector_kwargs = dict(return_features=True)
    metricname = None

    real_features = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_all=True, max_items=max_real).get_all_torch().to(torch.float16).to(opts.device)

    gen_features = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_all=True, max_items=num_gen).get_all_torch().to(torch.float16).to(opts.device)

    results = dict()
    for name, manifold, probes in [('precision', real_features, gen_features), ('recall', gen_features, real_features)]:
        kth = []
        for manifold_batch in manifold.split(row_batch_size):
            dist = compute_distances(row_features=manifold_batch, col_features=manifold, num_gpus=opts.num_gpus, rank=opts.rank, col_batch_size=col_batch_size)
            kth.append(dist.to(torch.float32).kthvalue(nhood_size + 1).values.to(torch.float16) if opts.rank == 0 else None)
        kth = torch.cat(kth) if opts.rank == 0 else None
        pred = []
        for probes_batch in probes.split(row_batch_size):
            dist = compute_distances(row_features=probes_batch, col_features=manifold, num_gpus=opts.num_gpus, rank=opts.rank, col_batch_size=col_batch_size)
            pred.append((dist <= kth).any(dim=1) if opts.rank == 0 else None)
        results[name] = float(torch.cat(pred).to(torch.float32).mean() if opts.rank == 0 else 'nan')
    return results['precision'], results['recall']


#----------------------------------------------------------------------------


_feature_detector_cache = dict()

def get_feature_detector_name(url,metricname=None):
    if url is None:
        return metricname
    else:
        return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

def compute_nn(opts, max_real, num_gen, nhood_size):
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(opts.device)
    batch_gen = 1
    
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    num_items = len(dataset)
    batch_size = 16
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    loader =  torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, 
                                          shuffle=False, **data_loader_kwargs)
       
    # Setup generator and load labels.
    G = opts.G.eval().requires_grad_(False).to(opts.device)
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    # Image generation func.
    def run_generator(z, c):
        img = G(z=z, c=c, **opts.G_kwargs)
        return torch.clamp(img , -1.,1.)
    
    
    images = []
    with torch.no_grad():
        for _i in range(num_gen // batch_gen):
            z = torch.randn([batch_gen, G.z_dim], device=opts.device)
            img = run_generator(z, None)
            real_images = []
            dist = [] 
            for i, (real_img, label) in enumerate(loader):
                real_img = (real_img.to(opts.device).to(torch.float32) / 127.5 - 1)
                real_images.append(real_img)
                dist.append(loss_fn_vgg(img.expand(real_img.size(0),3,256,256), real_img).squeeze())
                
            dist = torch.cat(dist, 0)
            topk = torch.topk(-dist, 10)[1]
            real_images = torch.cat(real_images, 0)
            real_images = real_images[topk]
            images.append(torch.cat([img, real_images],0))
           
    return torch.cat( images, 0)


#----------------------------------------------------------------------------

