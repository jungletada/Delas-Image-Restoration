import torch
import torch.nn as nn


class DelasGenerator(nn.Module):
    def __init__(self, netG, netA):
        super(DelasGenerator, self).__init__()
        self.netG = netG
        self.netA = netA


    def forward(self, L, N):
        y1, P = self.netG.forward(torch.cat((L, N), dim=1))
        y2, U = self.netA.forward(y1, N, P)
        return  y2, U
