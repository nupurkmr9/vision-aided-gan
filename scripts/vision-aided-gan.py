import sys
sys.path.append('../vision-aided-gan/')
from model_selection import *
import subprocess
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser("vision-aided-gan",)
    parser.add_argument("--cmd", default='', type=str, help="python command for vision-aided-gan training")
    parser.add_argument("--cv-args", default='', type=str, help="python command for vision model related args in vision-aided-gan training")
    parser.add_argument("--kimgs-list", default='4000,1000,1000', type=str, help="comma separated iterations to train with each pretrained model")
    parser.add_argument("--num", default=3, type=int, help="number of models in vision-aided-gan training")
    return parser.parse_args()

cv_loss_dict = {
    'clip': 'multilevel',
    'dino': 'multilevel',
    'swin': 'sigmoid',
    'vgg': 'sigmoid',
    'seg_ade': 'sigmoid',
    'det_coco': 'sigmoid',
    'face_normals': 'sigmoid',
    'face_seg': 'sigmod',
}

def launch(cmd, cv_args, kimgs_list, num_models):
    cv_models_list = [
        'input-swin-output-pool',
        'input-clip-output-pool',
        'input-dino-output-pool',
        'input-vgg-output-pool',
        'input-seg_ade-output-feat_pool',
        'input-det_coco-output-object_feat_pool',
        'input-face_parsing-output-pool',
        'input-face_normals-output-pool',
    ]
    
    ###########################################
    #### training with first cv model #########
    ###########################################
    data = cmd.split('--data')[1].split()[0]
    batch = int(cmd.split('--batch')[1].split()[0])
    outdir = cmd.split('--outdir')[1].split()[0]
    if '--resume' not in cmd:
        ### run baseline training for first 0.5M iterations ####
        prev_run_dirs = []
        if os.path.isdir(outdir):
            prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        run_cmd = cmd + ' --kimg 500'
        subprocess.call(run_cmd, shell=True)
        resumefile = os.path.join(glob.glob(os.path.join(outdir, f'{cur_run_id:05d}*'))[0],  'network-snapshot-000500.pkl')
    else:
        resumefile = cmd.split('--resume')[1].split()[0]
        cmd = cmd.replace('--resume '+ resumefile, '')
    
    #### model selection ####
    model_name, acc, network_pkl = calc_linearprobe(resumefile, data, batch, cv_models_list, torch.device('cuda', 0))
    cv_models_list.remove(model_name)
    
    cv1 = model_name.split('input-')[1].split('-output')[0]
    cv_loss = cv_loss_dict[cv1]
    cv = 'input-'+cv1+'-output-conv_multi_level'
    if acc > 0.9 or 'diffaugment' in augcv:
        cv_loss += '_s'
    run_cmd = cmd + ' --resume ' + network_pkl + ' ' + cv_args + ' --cv-loss ' + cv_loss + ' --cv '+ cv + ' --kimg ' + kimgs_list.split(',')[0] 

    ### get model directory of training run ####
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    
    ### launch training ####
    subprocess.call(run_cmd, shell=True)
    
    ####################################################
    #### training with second and more cv models #######
    ####################################################
    cv_used = [cv1]
    
    for num in range(num_models-1):
        #### model selection ####
        run_dir = glob.glob(os.path.join(outdir, f'{cur_run_id:05d}*'))[0]
        print("$$$$$$$$$", run_dir)
        model_name, acc, network_pkl = calc_linearprobe(run_dir, data, batch, cv_models_list, torch.device('cuda', 0))
        cv_models_list.remove(model_name)
        
        cv_next = model_name.split('input-')[1].split('-output')[0]
        cv_used.append(cv_next)
        cv_loss += ','+cv_loss_dict[cv_next]
        cv = 'input-'+ '-'.join(cv_used) +'-output-conv_multi_level_list'+str(num+2)
        if acc > 0.9 or 'diffaugment' in augcv:
            cv_loss += '_s'

        run_cmd = cmd + ' --resume ' + network_pkl + ' ' + cv_args + ' --cv-loss ' + cv_loss + ' --cv '+ cv + ' --exact-resume 2 --kimg ' + kimgs_list.split(',')[num+1]
        
        ### get model directory of training run ####
        prev_run_dirs = []
        if os.path.isdir(outdir):
            prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        
        ### launch training ####
        subprocess.call(run_cmd, shell=True)
                        
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    model_name = launch(cmd=args.cmd, cv_args=args.cv_args, kimgs_list=args.kimgs_list, num_models=args.num)