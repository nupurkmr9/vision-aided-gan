import torch
import torch.nn as nn
from torch_utils import persistence
from training.networks import Conv2dLayer
from training.networks import FullyConnectedLayer


@persistence.persistent_class
class MultiLevelDViT(nn.Module):
    def __init__(self, level=3, in_ch1=768, in_ch2=512, out_ch=256, down=1):
        super().__init__()

        self.decoder1 = []
        for i in range(level-1):
            self.decoder1.append(nn.Sequential(Conv2dLayer(in_ch1, out_ch, kernel_size =3, down=down, activation='lrelu'), 
                                                Conv2dLayer(out_ch, 1, kernel_size=1, down=2)))

        self.decoder1.append(nn.Sequential(FullyConnectedLayer(in_ch2, out_ch, activation='lrelu'),
                                                  FullyConnectedLayer(out_ch, 1)))

        self.decoder1 = nn.ModuleList(self.decoder1)

    def forward(self, input):

        final_pred = []

        for i in range(len(input)-1):
            final_pred.append(self.decoder1[i](input[i]).reshape(-1, 1, 9))

        final_pred.append(self.decoder1[-1](input[-1].float()))

        return final_pred


@persistence.persistent_class
class SimpleD(nn.Module):
    def __init__(self, in_ch=768, out_ch=256, out_size=3):
        super().__init__()

        self.decoder = nn.Sequential(Conv2dLayer(in_ch, out_ch, kernel_size=3, down=2, activation='lrelu'),
                                        nn.Flatten(),
                                        FullyConnectedLayer(out_ch*out_size*out_size, 256, activation='lrelu'),
                                        FullyConnectedLayer(out_ch, 1))

    def forward(self, input):

        return self.decoder(input)


@persistence.persistent_class
class MLPD(nn.Module):
    def __init__(self, in_ch=768, out_ch=256):
        super().__init__()

        self.decoder = nn.Sequential(FullyConnectedLayer(in_ch, out_ch, activation='lrelu'),
                                                  FullyConnectedLayer(out_ch, 1))

    def forward(self, input):
        return self.decoder(input)


@persistence.persistent_class
class MultiLevelDConv(nn.Module):
    def __init__(self, level=3, in_ch1=512, in_ch2=1024, in_ch3=2048, out_ch=128):
        super().__init__()

        self.decoder1 = []
        activation = 'lrelu'

        self.decoder1.append(nn.Sequential(
                    Conv2dLayer(in_ch1, out_ch, kernel_size=3, down=2, activation='lrelu'),
                    Conv2dLayer(in_ch1, 1, kernel_size=1, down=2)))

        self.decoder1.append(nn.Sequential(
                    Conv2dLayer(in_ch2, out_ch, kernel_size=3, activation=activation),
                    Conv2dLayer(out_ch, 1, kernel_size=1, down=2)))

        self.decoder1.append(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                            FullyConnectedLayer(in_ch3, out_ch, activation='lrelu'),
                                            FullyConnectedLayer(out_ch, 1)))

        self.decoder1 = nn.ModuleList(self.decoder1)

    def forward(self, input):

        final_pred = []

        final_pred.append(self.decoder1[0](input[0].float()).reshape(-1, 1, 49))
        final_pred.append(self.decoder1[1](input[1].float()).reshape(-1, 1, 49))

        final_pred.append(self.decoder1[-1](input[-1].float()))

        return final_pred


@persistence.persistent_class
class DiscriminatorAux(torch.nn.Module):
    def __init__(self, cv_type):
        super().__init__()

        def get_decoder(decoder_type, cv_type):
            if 'clip' in decoder_type:
                if 'conv_multi_level' in cv_type:
                    decoder = MultiLevelDViT(level=3, in_ch1=768, in_ch2=512, out_ch=256)
                else:
                    decoder = MLPD(in_ch=512, out_ch=256)

            if 'swin' in decoder_type:
                decoder = SimpleD(in_ch=768, out_ch=256)

            if 'dino' in decoder_type:
                if 'conv_multi_level' in cv_type:
                    decoder = MultiLevelDViT(level=3, in_ch1=768, in_ch2=768, out_ch=128, down=2)
                else:
                    decoder = MLPD(in_ch=768, out_ch=256)

            if 'vgg' in decoder_type:
                decoder = SimpleD(in_ch=512, out_ch=256)

            if 'seg' in decoder_type:
                if 'face' in cv_type:
                    decoder = SimpleD(in_ch=256, out_ch=256, out_size=4)

                elif 'ade' in cv_type:
                    decoder = SimpleD(in_ch=768, out_ch=256, out_size=4)

            if 'det' in decoder_type:
                decoder = SimpleD(in_ch=768, out_ch=256, out_size=4)

            if 'normals' in decoder_type:
                decoder = SimpleD(in_ch=512, out_ch=256, out_size=4)

            return decoder

        self.decoder = []
        cv_lists = cv_type.split('input-')[1].split('-output-')[0].split('-')
        for each in cv_lists:
            self.decoder.append(get_decoder(each, cv_type))

        self.decoder = nn.ModuleList(self.decoder)

    def forward(self, cv_feat, c):

        if len(self.decoder) == 1:
            pred_mask = self.decoder[0](cv_feat)
        else:
            pred_mask = []
            for i, each in enumerate(cv_feat):
                pred_mask.append(self.decoder[i](each))

        return pred_mask
