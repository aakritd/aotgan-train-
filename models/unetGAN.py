from .unet_generator import UnetGenerator
from torch import nn
import torch
from .base import BaseNetwork
from torch.nn.utils import spectral_norm

class Generator(BaseNetwork):
    def __init__(self):
        super(Generator, self).__init__()
        self.generator = UnetGenerator()

        self.init_weights()
    
    def forward(self, x):
        
        return self.generator(x)



# netG = Generator()
# inp = torch.rand(1, 4, 512, 512)
# print(netG(inp).shape)


class Discriminator(BaseNetwork):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(

            #Layer 1
            spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace = True),

            #Layer 2
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace = True),

            #Layer 3
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace = True),

            #Layer 4
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace = True),

            #Layer 5
            nn.Conv2d(512, 1, 4, 1, 1),
            
        )

        self.init_weights()
    
    def forward(self,x):
        output = self.discriminator(x)
        return output

# netD = Discriminator()
# x = torch.rand(1,3,512,512)
# print(netD(x).shape)


# '''
# Why do we use Spectral Normalization in GANs?
# '''