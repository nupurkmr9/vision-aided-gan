import numpy as np
from sklearn.linear_model import LogisticRegression
import torch

import utils_biggan as utils
from vision_aided_module import cvmodel
import dnnlib


def calc_linearprobe(config):
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
    G.load_state_dict(torch.load(dnnlib.util.open_file_or_url(config['network'])))
    G.eval()
    gen = lambda z: (G(z, G.shared(torch.randint(0, config['n_classes'], (z.size(0),), device=z.device))))

    cv_models_list = [
            'input-swin-output-pool',
            'input-clip-output-pool',
            'input-dino-output-pool',
            'input-vgg-output-pool',
            'input-seg_ade-output-pool',
            'input-det_coco-output-pool',
        ]

    metrics = {}

    for cv in cv_models_list:
        print(f'evaluating with {cv} features')
        output = cv.split('output-')[1]
        cv = cv.split('-output')[0].split('input-')[1]
        metrics[cv] = {}

        cv_pipe = cvmodel.CVBackbone(cv, output, diffaug=False).to(device)

        val_feats = []
        val_label = []
        feats = []
        label = []

        avg_train_accuracy = []
        avg_val_accuracy = []

        for _ in range(3):
            D_batch_size = (config['batch_size'] * config['num_D_steps']
                            * config['num_D_accumulations'])
            batch = D_batch_size
            training_set_iterator = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                                           'start_itr': 0})[0]

            feats = []
            label = []
            print(len(training_set_iterator))
            with torch.no_grad():
                for _, (val_img, _) in enumerate(training_set_iterator):
                    val_img = val_img.to(device)
                    val_feat = cv_pipe([val_img])[0]
                    feats.append(val_feat.cpu())
                    label.append(np.ones(val_feat.size(0)))
                    if len(feats)*batch > 10000:
                        break

            feats = torch.cat(feats, 0)
            print(feats.size())
            label = np.concatenate(label, axis=0)

            shuffle = np.random.permutation(feats.size(0))
            feats = feats[shuffle]
            train_feats, val_feats = feats.chunk(2)
            train_label, val_label = label[:train_feats.size(0)], label[train_feats.size(0):]
            num_val_images = val_label.shape[0]

            feats = []
            label = []

            with torch.no_grad():
                for _ in range(len(training_set_iterator)):
                    z = torch.randn(batch, config['dim_z'])
                    val_img = gen(z.to(device))
                    val_feat = cv_pipe([val_img])[0]
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

            max_iter = 1000
            shuffle = np.random.permutation(feats.shape[0])

            clf = LogisticRegression(random_state=0, max_iter=max_iter, C=0.2).fit(feats[shuffle], label[shuffle])
            avg_train_accuracy.append(clf.score(feats, label))
            avg_val_accuracy.append(clf.score(val_feats, val_label))

        print("train score:", np.mean(avg_train_accuracy), np.std(avg_train_accuracy))

        print(num_val_images, "val score:", np.mean(avg_val_accuracy), np.std(avg_val_accuracy),
                clf.predict_proba(val_feats[:num_val_images])[:, 1].mean(),
                clf.predict_proba(val_feats[num_val_images:])[:, 0].mean())

        metrics[cv]['linear'] = {'train': np.mean(avg_train_accuracy), 'train_std': np.std(avg_train_accuracy),
                                 'val': np.mean(avg_val_accuracy), 'val_std': np.std(avg_val_accuracy)
                                 }

    linear_probe_acc = [metrics[x]['linear']['val'] for x in metrics.keys()]
    cv_models_list = list(metrics.keys())

    print(metrics)

    print("selected model is:", cv_models_list[np.argmax(linear_probe_acc)], np.max(linear_probe_acc))
    return cv_models_list[np.argmax(linear_probe_acc)], np.max(linear_probe_acc), config['network']

# ----------------------------------------------------------------------------


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    calc_linearprobe(config)


if __name__ == '__main__':
    main()

