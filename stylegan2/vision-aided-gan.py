import os
import argparse
import re
import glob
import torch

import train
from model_selection import calc_linearprobe

cv_loss_dict = {
    'clip': 'multilevel_sigmoid',
    'dino': 'multilevel_sigmoid',
    'swin': 'sigmoid',
    'vgg': 'sigmoid',
    'seg_ade': 'sigmoid',
    'det_coco': 'sigmoid',
    'face_normals': 'sigmoid',
    'face_seg': 'sigmod',
}
cv_outputs = {
    'clip': 'conv_multi_level',
    'dino': 'conv_multi_level',
    'swin': 'conv',
    'vgg': 'conv',
    'seg_ade': 'conv',
    'det_coco': 'conv',
    'face_normals': 'conv',
    'face_seg': 'conv',
}

cv_models_list = [
    'input-swin-output-pool',
    'input-clip-output-pool',
    'input-dino-output-pool',
    'input-vgg-output-pool',
    'input-seg_ade-output-pool',
    'input-det_coco-output-pool',
    'input-face_seg-output-pool',
    'input-face_normals-output-pool',
    ]


def parse_args():
    parser = argparse.ArgumentParser("vision-aided-gan",  parents=[train.get_args_parser()], add_help=False)
    parser.add_argument("--kimgs-list", default='4000,1000,1000', type=str, help="comma separated iterations to train with each pretrained model")
    parser.add_argument("--num", default=3, type=int, help="number of models in vision-aided-gan training: maximum 3")
    return parser.parse_args()


def launch(args):
    torch.multiprocessing.set_start_method('spawn')
    data = args.data
    batch = args.batch
    outdir = args.outdir

    if args.resume is None:
        # run baseline training for first 0.5M iterations ####
        prev_run_dirs = []
        if os.path.isdir(outdir):
            prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        args.kimg = 500
        train.main(args)
        resumefile = os.path.join(glob.glob(os.path.join(outdir, f'{cur_run_id:05d}*'))[0],  'network-snapshot-000500.pkl')
    else:
        resumefile = args.resume

    ######################################################################################
    # training vision-aided-gan model progressive addition of pretrained models #########
    ######################################################################################

    cv_used = []
    for i in range(args.num):
        # model selection ####
        cv_model, acc, network_pkl = calc_linearprobe(resumefile, data, batch, cv_models_list, torch.device('cuda', 0))
        cv_models_list.remove(cv_model)

        # initialize arguments for vision-aided training ####
        cv_model = cv_model.split('input-')[1].split('-output')[0]
        cv_used.append(cv_model)
        args.cv = 'input-'+'+'.join([x for x in cv_used])+'-output-'+'+'.join([cv_outputs[x] for x in cv_used])

        if i > 0:
            args.cv_loss += '+'

        args.cv_loss += cv_loss_dict[cv_model]
        if acc > 0.9 or 'diffaugment' in args.augcv:
            args.cv_loss += '_s'

        args.resume = network_pkl
        args.kimg = int(args.kimgs_list.split(',')[i])

        # get model directory of training run ####
        prev_run_dirs = []
        if os.path.isdir(outdir):
            prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1

        # launch training ####
        torch.cuda.empty_cache()
        train.main(args)
        resumefile = os.path.join(glob.glob(os.path.join(outdir, f'{cur_run_id:05d}*'))[0], 'network-snapshot-best.pkl')

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    args = parse_args()
    model_name = launch(args)
