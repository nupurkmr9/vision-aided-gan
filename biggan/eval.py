import torch
from tqdm import tqdm
import numpy as np
import functools
import inception_utils
import utils_biggan as utils
import dnnlib

from cleanfid import fid


def run_eval(config):
    # update config (see train.py for explanation)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    config = utils.update_config_roots(config)
    config['skip_init'] = True
    config['no_optim'] = True
    device = 'cuda'

    model = __import__(config['model'])
    G = model.Generator(**config).cuda()
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                               device=device, fp16=config['G_fp16'],
                               z_var=config['z_var'])

    G.load_state_dict(torch.load(dnnlib.util.open_file_or_url(config['network'])))

    if config['G_eval_mode']:
        G.eval()
    else:
        G.train()

    clean = True
    if clean:
        gen = lambda z: ((G(z, G.shared(torch.randint(0, config['n_classes'], (z.size(0),), device=z.device)))*0.5+0.5)*255).clamp(0, 255).to(torch.uint8)
        print(gen(torch.randn(2, 128).cuda()).size())
        FID_list = []
        for _ in tqdm(range(config['repeat'])):
            FID = fid.compute_fid(gen=gen, z_dim=128, num_gen=10000, dataset_name=config['dataset_name'], dataset_res=128,  mode="clean", dataset_split="custom", batch_size=32)
            FID_list.append(FID)
        if config['repeat'] > 1:
            print('FID mean: {}, std: {}'.format(np.mean(FID_list), np.std(FID_list)))
        else:
            print('FID: {}'.format(np.mean(FID_list)))
    else:
        get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'], config['no_fid'])
        sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)
        IS_list = []
        FID_list = []
        for _ in tqdm(range(config['repeat'])):
            IS, _, FID = get_inception_metrics(sample, config['num_inception_images'], num_splits=10, prints=False)
            IS_list.append(IS)
            FID_list.append(FID)
        if config['repeat'] > 1:
            print('IS mean: {}, std: {}'.format(np.mean(IS_list), np.std(IS_list)))
            print('FID mean: {}, std: {}'.format(np.mean(FID_list), np.std(FID_list)))
        else:
            print('IS: {}'.format(np.mean(IS_list)))
            print('FID: {}'.format(np.mean(FID_list)))


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    run_eval(config)


if __name__ == '__main__':
    main()
