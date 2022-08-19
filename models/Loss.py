import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_cuda):
        super(GANLoss, self).__init__()
        r = np.random.randint(9, 10)
        f = np.random.randint(0, 1)
        self.real_label = r /10.
        self.fake_label = f / 10.
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        if use_cuda:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.FloatTensor

    def get_target_tensor(self, target, target_is_real):
        if target_is_real:  # real image
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(target)

        else:              # fake image
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(target)

    def get_zero_tensor(self, target):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(target)

    def loss(self, target, target_is_real):
        # cross entropy loss
        target_tensor = self.get_target_tensor(target, target_is_real)
        loss = F.binary_cross_entropy_with_logits(target, target_tensor)
        return loss

    def __call__(self, target, target_is_real):
        # |input| may not be a tensor, but list of tensors in case of multi-scale discriminator
        if isinstance(target, list):
            loss = 0
            for pred_i in target:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(target)
        else:
            return self.loss(target, target_is_real)
