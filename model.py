import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.models.vgg as vgg

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2

    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    og = np.ogrid[:kernel_size, :kernel_size]

    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt

    return torch.from_numpy(weight).float()

class segmentation_model(nn.Module):
    def __init__(self, n_class):
        super(segmentation_model, self).__init__()
        vgg_model = models.vgg16(pretrained=True)
        self.features = vgg_model.features
        
        self.encoder1 = self.features[:5]
        self.encoder2 = self.features[5:10]
        self.encoder3 = self.features[10:17]
        self.encoder4 = self.features[17:24]
        self.encoder5 = self.features[24:31]
        
        self.decoder4 = self.upconv(512, 512)
        self.decoder3 = self.upconv(512, 256)
        self.decoder2 = self.upconv(256, 128)
        self.decoder1 = self.upconv(128, 64)
        
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(64, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(2048, 4096, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(4096, 2048, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ConvTranspose2d(256, n_class, kernel_size=64, stride=32, padding=16, output_padding=0, bias=False),
        )
        # self._initialize_weights()

    def upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        d4 = self.decoder4(e5)
        d3 = self.decoder3(d4 + e4)
        d2 = self.decoder2(d3 + e3)
        d1 = self.decoder1(d2 + e2)
        
        out = self.segmentation_head(d1 + e1)
        out = nn.functional.interpolate(out, size=(132, 132), mode='bilinear', align_corners=True)
        
        return out
