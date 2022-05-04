import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


class sigmoid_loss(torch.nn.Module):
    def __init__(self, alpha=1.):
        super().__init__()
        self.lossfn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, input, for_real=True, for_G=False):
        if for_G:
            for_real = True
        if for_real:
            target = self.alpha*torch.tensor(1.)
        else:
            target = torch.tensor(0.)

        target_ = target.expand_as(input).to(input.device)
        loss = self.lossfn(input, target_).mean(1).reshape(-1, 1)
        return loss


class multilevel_loss(torch.nn.Module):
    def __init__(self, alpha=1.):
        super().__init__()
        self.lossfn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, input, for_real=True, for_G=False):
        if for_G:
            for_real = True
        if for_real:
            target = self.alpha*torch.tensor(1.)
        else:
            target = torch.tensor(0.)

        loss = 0
        for _, each in enumerate(input):
            target_ = target.expand_as(each).to(each.device)
            loss_ = self.lossfn(each, target_)
            if len(loss_.size()) > 2:
                loss_ = loss_.mean([1, 2]).reshape(-1, 1)
            loss += loss_
        return loss


class hinge_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, for_real=True, for_G=False):
        if for_G:
            loss = -torch.mean(input)
        else:
            if for_real:
                loss = torch.mean(F.relu(1. - input))
            else:
                loss = torch.mean(F.relu(1. + input))
                    
        return loss


class multilevel_hinge_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, for_real=True, for_G=False):
        loss = 0
        for _, each in enumerate(input):
            if for_G:
                loss_ = -torch.mean(each)
            else:
                if for_real:
                    loss_ = torch.mean(F.relu(1. - each))
                else:
                    loss_ = torch.mean(F.relu(1. + each))
            loss += loss_
        return loss


loss_dict = {
    'sigmoid': sigmoid_loss,
    'sigmoid_s': functools.partial(sigmoid_loss, alpha=0.8),
    'multilevel_sigmoid': multilevel_loss,
    'multilevel_sigmoid_s': functools.partial(multilevel_loss, alpha=0.8),
    'hinge': hinge_loss,
    'multilevel_hinge': multilevel_hinge_loss,
}


class losses_list(torch.nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.losses = []
        for each in loss_type.split('+'):
            self.losses.append(loss_dict[each]())

    def forward(self, input, **kwargs):
        loss = 0
        for i in range(len(self.losses)):
            loss_ = self.losses[i](input[i], **kwargs)
            loss += loss_
        return loss
