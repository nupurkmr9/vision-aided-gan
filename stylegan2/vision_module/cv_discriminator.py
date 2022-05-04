import torch
import torch.nn as nn

from vision_module.cvmodel import CVBackbone
from training.networks import Conv2dLayer
from training.networks import FullyConnectedLayer


class MultiLevelDViT(nn.Module):
    def __init__(self, level=3, in_ch1=768, in_ch2=512, out_ch=256, down=1):
        super().__init__()

        self.decoder = nn.ModuleList()
        for _ in range(level-1):
            self.decoder.append(nn.Sequential(Conv2dLayer(in_ch1, out_ch, kernel_size=3, down=down, activation='lrelu'),
                                              Conv2dLayer(out_ch, 1, kernel_size=1, down=2)))
        self.decoder.append(nn.Sequential(FullyConnectedLayer(in_ch2, out_ch, activation='lrelu'),
                                          FullyConnectedLayer(out_ch, 1)))

    def forward(self, input):
        final_pred = []
        for i in range(len(input)-1):
            final_pred.append(self.decoder[i](input[i]).squeeze(1))
        final_pred.append(self.decoder[-1](input[-1].float()))

        return final_pred


class SimpleD(nn.Module):
    def __init__(self, in_ch=768, out_ch=256, out_size=3):
        super().__init__()

        self.decoder = nn.Sequential(Conv2dLayer(in_ch, out_ch, kernel_size=3, down=2, activation='lrelu'),
                                     nn.Flatten(),
                                     FullyConnectedLayer(out_ch*out_size*out_size, out_ch, activation='lrelu'),
                                     FullyConnectedLayer(out_ch, 1))

    def forward(self, input):

        return self.decoder(input)


class MLPD(nn.Module):
    def __init__(self, in_ch=768, out_ch=256):
        super().__init__()

        self.decoder = nn.Sequential(FullyConnectedLayer(in_ch, out_ch, activation='lrelu'),
                                     FullyConnectedLayer(out_ch, 1))

    def forward(self, input):
        return self.decoder(input)


class Discriminator(torch.nn.Module):
    def __init__(self, cv_type, output_type='conv_multi_level', diffaug=False, device='cpu'):
        super().__init__()

        self.cv_ensemble = CVBackbone(cv_type, output_type, diffaug=diffaug, device=device)
        self.num_models = len(self.cv_ensemble.models)

        def get_decoder(cv_type, output_type):
            if 'clip' in cv_type:
                if 'conv_multi_level' in output_type:
                    decoder = MultiLevelDViT(level=3, in_ch1=768, in_ch2=512, out_ch=256)
                else:
                    decoder = MLPD(in_ch=512, out_ch=256)

            if 'swin' in cv_type:
                decoder = SimpleD(in_ch=768, out_ch=256)

            if 'dino' in cv_type:
                if 'conv_multi_level' in output_type:
                    decoder = MultiLevelDViT(level=3, in_ch1=768, in_ch2=768, out_ch=128, down=2)
                else:
                    decoder = MLPD(in_ch=768, out_ch=256)

            if 'vgg' in cv_type:
                decoder = SimpleD(in_ch=512, out_ch=256)

            if 'seg' in cv_type:
                if 'face' in cv_type:
                    decoder = SimpleD(in_ch=256, out_ch=256, out_size=4)
                elif 'ade' in cv_type:
                    decoder = SimpleD(in_ch=768, out_ch=256, out_size=4)

            if 'det' in cv_type:
                decoder = SimpleD(in_ch=768, out_ch=256, out_size=4)

            if 'normals' in cv_type:
                decoder = SimpleD(in_ch=512, out_ch=256, out_size=4)

            return decoder

        self.decoder = nn.ModuleList()
        cv_type = cv_type.split('+')
        output_type = output_type.split('+')

        for cv_type_, output_type_ in zip(cv_type, output_type):
            self.decoder.append(get_decoder(cv_type_, output_type_))

    def train(self, mode=True):
        self.cv_ensemble = self.cv_ensemble.train(False)
        self.decoder = self.decoder.train(mode)
        return self

    def forward(self, images, c=None, detach=False):
        if detach:
            with torch.no_grad():
                cv_feat = self.cv_ensemble(images)
        else:
            cv_feat = self.cv_ensemble(images)

        pred_mask = []
        for i, x in enumerate(cv_feat):
            pred_mask.append(self.decoder[i](x))
        return pred_mask
