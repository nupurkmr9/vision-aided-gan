import torch
import torch.nn as nn
from torch_utils import training_stats


class sigmoid_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def loss(self, input, for_real=None):
        if for_real:
            loss = torch.nn.functional.softplus(-input)
        else:
            loss = torch.nn.functional.softplus(input)
        return loss.mean()


class multilevel_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lossfn = nn.BCEWithLogitsLoss(reduction='none')

    def loss(self, input, for_real=True):
        if for_real:
            target = torch.tensor(1.)
        else:
            target = torch.tensor(0.)

        loss = 0
        for _, each in enumerate(input):
            target_ = target.expand_as(each).to(each.device)
            loss_ = self.lossfn(each, target_)
            if len(loss_.size()) > 2:
                loss_ = loss_.mean(2).reshape(-1, 1)

            loss += loss_
        return loss.mean()


class sigmoid_loss_smooth(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lossfn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = 0.8

    def loss(self, input, for_real=None):
        if for_real:
            target = self.alpha*torch.tensor(1.)
        else:
            target = torch.tensor(0.)

        target_ = target.expand_as(input).to(input.device)
        loss = self.lossfn(input, target_)
        return loss.mean()


class multilevel_loss_smooth(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.8
        self.lossfn = nn.BCEWithLogitsLoss(reduction='none')

    def loss(self, input, for_real=True):
        if for_real:
            target = self.alpha*torch.tensor(1.)
        else:
            target = torch.tensor(0.)

        loss = 0
        for _, each in enumerate(input):
            target_ = target.expand_as(each).to(each.device)
            loss_ = self.lossfn(each, target_)
            if len(loss_.size()) > 2:
                loss_ = loss_.mean(2).reshape(-1, 1)

            loss += loss_

        return loss.mean()


loss_dict = {
    'sigmoid': sigmoid_loss,
    'sigmoid_s': sigmoid_loss_smooth,
    'multilevel': multilevel_loss,
    'multilevel_s': multilevel_loss_smooth,
}


class losses_list(torch.nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.losses = []
        for each in loss_type.split(','):
            self.losses.append(loss_dict[each]())

    def loss(self, input, for_real, forG=False):
        loss = 0
        for i in range(len(self.losses)):
            loss_ = self.losses[i].loss(input[i], for_real=for_real)
            loss += loss_
            if forG:
                training_stats.report('Loss/G/vison_aided_loss' + str(i), loss_)
            elif for_real:
                training_stats.report('Loss/D/vision_aided_loss_real' + str(i), loss_)
            else:
                training_stats.report('Loss/D/vision_aided_loss_fake' + str(i), loss_)
        return loss
