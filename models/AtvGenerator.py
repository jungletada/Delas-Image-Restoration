import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGenerator(nn.Module):
    def __init__(self, in_nc=3, p_nc=64):

        super(AttentionGenerator, self).__init__()

        self.actvn = nn.LeakyReLU(0.35, False)
        self.softmax = nn.Softmax(dim=1)

        self.conv_img = nn.Conv2d(in_nc+p_nc, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv_noise = nn.Conv2d(1+p_nc, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.pool2 = nn.AdaptiveAvgPool2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d(32)
        self.conv_pool = nn.Conv2d(p_nc*3, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv_inter = nn.Conv2d(128, 15, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv_uc = nn.Conv2d(128, 5, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv_ucmap = nn.Conv2d(5, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)

    def forward(self, y1, N, P):
        x1 = torch.cat((y1, P), dim=1)
        x1 = self.actvn(self.conv_img(x1))
        x1 = x1 * P
        x2 = torch.cat((N, P), dim=1)
        x2 = self.actvn(self.conv_noise(x2))
        Pr = x1 + x2
        P2 = self.pool2(Pr)
        P2 = F.interpolate(P2, scale_factor=2, mode='nearest')
        P3 = self.pool3(Pr)
        P3 = F.interpolate(P3, scale_factor=4, mode='nearest')
        Pr = torch.cat((Pr, P2, P3), dim=1)
        Pr = self.actvn(self.conv_pool(Pr))

        att = self.softmax(self.conv_uc(Pr))
        img = self.conv_inter(Pr)

        att1 = att[:, 0:1, :, :].repeat(1, 3, 1, 1)
        att2 = att[:, 1:2, :, :].repeat(1, 3, 1, 1)
        att3 = att[:, 2:3, :, :].repeat(1, 3, 1, 1)
        att4 = att[:, 3:4, :, :].repeat(1, 3, 1, 1)
        att5 = att[:, 4:5, :, :].repeat(1, 3, 1, 1)

        img1 = img[:, 0:3, :, :] * att1
        img2 = img[:, 3:6, :, :] * att2
        img3 = img[:, 6:9, :, :] * att3
        img4 = img[:, 9:12, :, :] * att4
        img5 = img[:, 12:15, :, :] * att5

        img_f = nn.Tanh()(img1 + img2 + img3 + img4 + img5)
        uc_map = self.conv_ucmap(att).repeat(1, 3, 1, 1)
        return img_f, uc_map


# if __name__ == '__main__':
#
#     x = torch.ones(1, 3, 128, 128)
#     P = torch.ones(1, 64, 128, 128)
#     N = torch.ones(1, 1, 128, 128)
#     net = AttentionGenerator()
#     with torch.no_grad():
#         y, u = net(x, N, P)
#         print(y.size())
#         print(u.size())
