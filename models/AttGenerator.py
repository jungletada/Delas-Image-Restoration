import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

class AttenGenerator(nn.Module):
    def __init__(self):
        super(AttenGenerator, self).__init__()
        ngf, in_nc = 64, 3

        self.down1 = nn.Conv2d(in_nc, ngf, kernel_size=3, stride=1, padding=1)
        self.down2 = nn.Conv2d(in_nc, ngf, kernel_size=2, stride=2, padding=0)
        self.down3 = nn.Conv2d(in_nc, ngf, kernel_size=4, stride=4, padding=0)

        self.actvn = nn.LeakyReLU(0.35, False)

        self.up1 = nn.ConvTranspose2d(ngf, in_nc, kernel_size=3, stride=1, padding=1)
        self.up2 = nn.ConvTranspose2d(ngf, in_nc, kernel_size=2, stride=2, padding=0)
        self.up3 = nn.ConvTranspose2d(ngf, in_nc, kernel_size=4, stride=4, padding=0)

        self.conv_xc = nn.Conv2d(in_nc*3, 110, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_img = nn.Conv2d(110, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_at1 = nn.Conv2d(110, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_at2 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_out1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv_out2 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1, bias=False)   
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x): # x: original
        t1 = self.actvn(self.down1(x))
        t2 = self.actvn(self.down2(x))
        t3 = self.actvn(self.down3(x))

        r1 = self.up1(t1) * x
        r2 = self.up2(t2) * x
        r3 = self.up3(t3) * x

        xc = torch.cat((r1,r2,r3),dim=1)
        xc = self.conv_xc(xc)
        image = self.conv_img(xc)
        attention = self.conv_at1(xc)
        attention = self.softmax(attention)

        attention1 = attention[:, 0:1, :, :].repeat(1, 3, 1, 1)
        attention2 = attention[:, 1:2, :, :].repeat(1, 3, 1, 1)
        attention3 = attention[:, 2:3, :, :].repeat(1, 3, 1, 1)
        attention4 = attention[:, 3:4, :, :].repeat(1, 3, 1, 1)
        attention5 = attention[:, 4:5, :, :].repeat(1, 3, 1, 1)
        attention6 = attention[:, 5:6, :, :].repeat(1, 3, 1, 1)
        attention7 = attention[:, 6:7, :, :].repeat(1, 3, 1, 1)
        attention8 = attention[:, 7:8, :, :].repeat(1, 3, 1, 1)
        
        image1 = image[:, 0:3, :, :] * attention1
        image2 = image[:, 3:6, :, :]  * attention2
        image3 = image[:, 6:9, :, :] * attention3
        image4 = image[:, 9:12, :, :] * attention4
        image5 = image[:, 12:15, :, :] * attention5
        image6 = image[:, 15:18, :, :] * attention6
        image7 = image[:, 18:21, :, :] * attention7
        image8 = image[:, 21:24, :, :] * attention8

        out_img = image1 + image2 + image3 + image4 + image5 + image6 + \
            image6 + image7 + image8
        # out_img = self.conv_out1(out_img)
        # out_img = self.sigmoid(out_img)
        # uncertainty map generation
        uncertainty = self.conv_at2(attention)
        
        uncertainty = uncertainty.repeat(1, 3, 1, 1)
        return uncertainty, out_img
        # return image1, image2, image3, image4, image5, image6, \
        #     image6, image7, image8, attention1, attention2, attention3,\
        #     attention4, attention5, attention6,attention7, attention8, out_img




if __name__ == '__main__':
    with torch.no_grad():
        x = torch.ones(5, 3, 128, 128)
        netTest = AttenGenerator()
        u, o = netTest.forward(x)
        print(u.size())
        print(o.size())
