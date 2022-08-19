import torch.nn as nn
import torch.nn.functional as F

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, in_nc=6):
        super().__init__()
        kw, nf, padw, n_layers_D =4, 64, 2, 4
        sequence = [[nn.Conv2d(in_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.35, False)]]

        for n in range(1, n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == n_layers_D - 1 else 2
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw),
                          nn.LeakyReLU(0.2, False)]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, x):
        results = [x]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        return results[1:]


