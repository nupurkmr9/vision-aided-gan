import torch
import os
import glob
import json
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
import dnnlib
import legacy

import vision_module

def parse_args():
    parser = argparse.ArgumentParser("vision-aided-gan",)
    parser.add_argument("--network", type=str, help='Network pickle filename or folder name if best model needs to be selected', required=True)
    parser.add_argument("--data", type=str, help='Real sample dataset for model selection (directory or zip) [same as training data]', required=True)
    parser.add_argument("--batch", type=int, default=64, help='Override batch size')
    return parser.parse_args()


def calc_linearprobe(network_pkl, data, batch=64, cv_models_list=None, device='cuda'):
    dnnlib.util.Logger(should_flush=True)

    if '.pkl' not in network_pkl:  # rundir given and get the best model from that rundir
        metricfile = glob.glob(os.path.join(network_pkl, 'metric-fid50k_full.jsonl'))[0]
        metric = open(metricfile, 'r')
        metric = metric.readlines()
        best_snapshot_fid = 999.
        for each in metric:
            if each.strip() == '':
                continue
            each = json.loads(each)
            fid = each["results"]["fid50k_full"]
            if float(fid) < best_snapshot_fid:
                best_snapshot_fid = float(fid)
                best_snapshot_pkl = each["snapshot_pkl"]

        network_pkl = os.path.join(network_pkl, best_snapshot_pkl)

    print("network file:", network_pkl)

    with dnnlib.util.open_url(network_pkl) as f:
        network_dict = legacy.load_network_pkl(f)
        G_ema = network_dict['G_ema']  # subclass of torch.nn.Module

    G_ema = G_ema.to(device)

    if cv_models_list is None:
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

    data_loader_kwargs = {'pin_memory': True, 'num_workers': 1, 'prefetch_factor': 2}

    metrics = {}

    for cv in cv_models_list:

        metrics[cv] = {}
        print("$$$$$$$$$$$$$$$$", cv, "$$$$$$$$$$$$$$$$")

        class_name = 'vision_module.cvmodel.CVBackbone'

        output_type = cv.split('output-')[1]
        cv_type = cv.split('-output')[0].split('input-')[1]
        
        cv_ensemble = vision_module.cvmodel.CVBackbone(cv_type= cv_type, output_type = output_type, diffaug = False).requires_grad_(False).to(device)

        val_feats = []
        val_label = []
        feats = []
        label = []

        avg_train_accuracy = []
        avg_val_accuracy = []

        for _ in range(3):

            training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset',
                                                  path=data, use_labels=True, max_size=None, xflip=False)
            training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)

            training_set_iterator = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch, **data_loader_kwargs)

            feats = []
            label = []
            with torch.no_grad():
                for _, (val_img, _) in enumerate(training_set_iterator):
                    val_img = (val_img.to(device).to(torch.float32) / 127.5 - 1.)
                    val_feat = cv_ensemble([val_img])[0]
                    feats.append(val_feat.cpu())
                    label.append(np.ones(val_feat.size(0)))
                    if len(feats)*batch > 10000:
                        break

            feats = torch.cat(feats, 0)
            label = np.concatenate(label, axis=0)

            shuffle = np.random.permutation(feats.size(0))
            feats = feats[shuffle]
            train_feats, val_feats = feats.chunk(2)
            train_label, val_label = label[:train_feats.size(0)], label[train_feats.size(0):]
            num_val_images = val_label.shape[0]

            feats = []
            label = []

            with torch.no_grad():
                for _ in range(len(training_set) // batch):
                    z = torch.randn(batch, G_ema.z_dim)
                    val_img = G_ema(z.to(device), None)
                    val_feat = cv_ensemble([val_img])[0]
                    feats.append(val_feat.cpu())
                    label.append(np.zeros(val_feat.size(0)))
                    if len(feats)*batch > 10000:
                        break

            feats = torch.cat(feats, 0)
            label = np.concatenate(label, axis=0)

            train_feats_fake, val_feats_fake = feats.chunk(2)
            train_label_fake, val_label_fake = label[:train_feats_fake.size(0)], label[train_feats_fake.size(0):]

            feats = torch.cat([train_feats, train_feats_fake], 0).cpu().numpy()
            label = np.concatenate([train_label, train_label_fake], 0)
            print("train set:", feats.shape, label.shape)

            val_feats = torch.cat([val_feats, val_feats_fake], 0).cpu().numpy()
            val_label = np.concatenate([val_label, val_label_fake], 0)
            print("val set:", val_feats.shape, val_label.shape)

            max_iter = 5000
            shuffle = np.random.permutation(feats.shape[0])

            clf = LogisticRegression(random_state=0, max_iter=max_iter, C=0.2).fit(feats[shuffle], label[shuffle])
            avg_train_accuracy.append(clf.score(feats, label))
            avg_val_accuracy.append(clf.score(val_feats, val_label))

        print("train score:", np.mean(avg_train_accuracy), np.std(avg_train_accuracy))

        print(num_val_images, "val score:", np.mean(avg_val_accuracy), np.std(avg_val_accuracy),)

        metrics[cv]['linear'] = {'train': np.mean(avg_train_accuracy), 'train_std': np.std(avg_train_accuracy),
                                 'val': np.mean(avg_val_accuracy), 'val_std': np.std(avg_val_accuracy)
                                 }

    linear_probe_acc = [metrics[x]['linear']['val'] for x in metrics.keys()]
    cv_models_list = list(metrics.keys())

    print(metrics)

    print("selected model is:", cv_models_list[np.argmax(linear_probe_acc)], np.max(linear_probe_acc))
    return cv_models_list[np.argmax(linear_probe_acc)], np.max(linear_probe_acc), network_pkl

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    args = parse_args()
    model_name, acc, network_pkl = calc_linearprobe(args.network, args.data, args.batch)
