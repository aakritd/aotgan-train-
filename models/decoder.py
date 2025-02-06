import torch
from torch import nn
import torch.nn.functional as F


class UpConv(nn.Module):
    def __init__(self, inc, ouc, k = 3, st = 1, pd = 1):
        super(UpConv, self).__init__()
        self.conv = nn.Conv2d(inc, ouc, k, st, pd)
    def forward(self, x):
        x = F.interpolate(x, scale_factor = 2, mode = "bilinear", align_corners = True)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(

            # Layer 1
            UpConv(256, 128),
            nn.ReLU(),

            # Layer 2
            UpConv(128,64),
            nn.ReLU(),

            # Layer 3
            nn.Conv2d(64,3,3,1,1),
            nn.Tanh()
        )
    

        
    def forward(self,x):
        output = self.decoder(x)
        return output




# decoder = Decoder()
# x = torch.rand(1,256,128,128)
# print(decoder(x).shape)