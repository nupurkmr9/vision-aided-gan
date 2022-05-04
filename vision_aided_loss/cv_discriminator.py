import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.nn import init

from vision_aided_loss.cvmodel import CVBackbone
from vision_aided_loss.blurpool import BlurPool
from vision_aided_loss.cv_losses import losses_list


class MultiLevelDViT(nn.Module):
    def __init__(self, level=3, in_ch1=768, in_ch2=512, out_ch=256, num_classes=0, activation=nn.LeakyReLU(0.2, inplace=True), down=1):
        super().__init__()

        self.decoder = nn.ModuleList()
        self.level = level
        for _ in range(level-1):
            self.decoder.append(nn.Sequential(
                                BlurPool(in_ch1, pad_type='zero', stride=1, pad_off=1) if down > 1 else nn.Identity(),
                                spectral_norm(nn.Conv2d(in_ch1, out_ch, kernel_size=3, stride=2 if down > 1 else 1, padding=1 if down == 1 else 0)),
                                activation,
                                BlurPool(out_ch, pad_type='zero', stride=1),
                                spectral_norm(nn.Conv2d(out_ch, 1, kernel_size=1, stride=2)))
                                )
        self.decoder.append(nn.Sequential(spectral_norm(nn.Linear(in_ch2, out_ch)), activation))
        self.out = spectral_norm(nn.Linear(out_ch, 1))
        self.embed = None
        if num_classes > 0:
            self.embed = nn.Embedding(num_classes, out_ch)                           

    def forward(self, x, c=None):

        final_pred = []
        for i in range(self.level-1):
            final_pred.append(self.decoder[i](x[i]).squeeze(1))

        h = self.decoder[-1](x[-1].float())
        out = self.out(h)

        if self.embed is not None:
            out += torch.sum(self.embed(c) * h, 1, keepdim=True)

        final_pred.append(out)
        # final_pred = torch.cat(final_pred, 1)
        return final_pred


class SimpleD(nn.Module):
    def __init__(self, in_ch=768, out_ch=256, out_size=3, num_classes=0, activation=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()

        self.decoder = nn.Sequential(
                                BlurPool(in_ch, pad_type='zero', stride=1, pad_off=1),
                                spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2)),
                                activation,
                                nn.Flatten(),
                                spectral_norm(nn.Linear(out_ch*out_size*out_size, out_ch)),
                                activation,)

        self.out = spectral_norm(nn.Linear(out_ch, 1))
        self.embed = None
        if num_classes > 0:
            self.embed = nn.Embedding(num_classes, out_ch)    

    def forward(self, x, c):
        h = self.decoder(x)
        out = self.out(h)
        if self.embed is not None:
            out += torch.sum(self.embed(c) * h, 1, keepdim=True)

        return out


class MLPD(nn.Module):
    def __init__(self, in_ch=768, out_ch=256, num_classes=0, activation=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        print(activation)
        self.decoder = nn.Sequential(spectral_norm(nn.Linear(in_ch, out_ch)),
                                     activation,
                                     )
        self.out = spectral_norm(nn.Linear(out_ch, 1))
        self.embed = None
        if num_classes > 0:
            self.embed = nn.Embedding(num_classes, out_ch)

    def forward(self, x, c):
        h = self.decoder(x)
        out = self.out(h)
        if self.embed is not None:
            out += torch.sum(self.embed(c) * h, 1, keepdim=True)
        return out


class Discriminator(torch.nn.Module):
    def __init__(self, cv_type, output_type='conv_multi_level', loss_type=None, diffaug=True, device='cpu', create_optim=False, num_classes=0, activation=nn.LeakyReLU(0.2, inplace=True), **kwargs):
        super().__init__()

        self.cv_ensemble = CVBackbone(cv_type, output_type, diffaug=diffaug, device=device)

        if loss_type is not None:
            self.loss_type = losses_list(loss_type=loss_type)
        else:
            self.loss_type = None

        self.num_models = len(self.cv_ensemble.models)

        def get_decoder(cv_type, output_type):
            if 'clip' in cv_type:
                if 'conv_multi_level' in output_type:
                    decoder = MultiLevelDViT(level=3, in_ch1=768, in_ch2=512, out_ch=256, num_classes=num_classes, activation=activation)
                else:
                    decoder = MLPD(in_ch=512, out_ch=256, num_classes=num_classes, activation=activation)

            if 'swin' in cv_type:
                decoder = SimpleD(in_ch=768, out_ch=256, num_classes=num_classes, activation=activation)

            if 'dino' in cv_type:
                if 'conv_multi_level' in output_type:
                    decoder = MultiLevelDViT(level=3, in_ch1=768, in_ch2=768, out_ch=128, down=2, num_classes=num_classes, activation=activation)
                else:
                    decoder = MLPD(in_ch=768, out_ch=256, num_classes=num_classes, activation=activation)

            if 'vgg' in cv_type:
                decoder = SimpleD(in_ch=512, out_ch=256, num_classes=num_classes, activation=activation)

            if 'seg' in cv_type:
                if 'face' in cv_type:
                    decoder = SimpleD(in_ch=256, out_ch=256, out_size=4, num_classes=num_classes, activation=activation)
                elif 'ade' in cv_type:
                    decoder = SimpleD(in_ch=768, out_ch=256, out_size=4, num_classes=num_classes, activation=activation)

            if 'det_coco' in cv_type:
                decoder = SimpleD(in_ch=768, out_ch=256, out_size=4, num_classes=num_classes, activation=activation)

            if 'normals' in cv_type:
                decoder = SimpleD(in_ch=512, out_ch=256, out_size=4, num_classes=num_classes, activation=activation)

            return decoder

        self.decoder = nn.ModuleList()
        cv_type = cv_type.split('+')
        output_type = output_type.split('+')

        for cv_type_, output_type_ in zip(cv_type, output_type):
            self.decoder.append(get_decoder(cv_type_, output_type_))

        if create_optim:
            self.init = kwargs['D_init']
            self.optim = torch.optim.Adam(params=self.decoder.parameters(), lr=kwargs['D_lr'],
                                          betas=(kwargs['D_B1'], kwargs['D_B2']), weight_decay=0, eps=kwargs['adam_eps'])

    def train(self, mode=True):
        self.cv_ensemble = self.cv_ensemble.train(False)
        self.decoder = self.decoder.train(mode)
        return self

    # Initialize (copied from BigGAN pytorch repo to support biggan code)
    def init_weights(self):
        self.param_count = 0
        for module in self.decoder.modules():
            if (isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
            self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for DAux''s initialized parameters: %d' % self.param_count)

    def forward(self, images, c=None, detach=False, **kwargs):
        if detach:
            with torch.no_grad():
                cv_feat = self.cv_ensemble(images)
        else:
            cv_feat = self.cv_ensemble(images)

        pred_mask = []
        for i, x in enumerate(cv_feat):
            pred_mask.append(self.decoder[i](x, c))

        if self.loss_type is not None:
            return self.loss_type(pred_mask, **kwargs)

        return pred_mask
