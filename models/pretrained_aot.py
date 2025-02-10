from .encoder import Encoder
from .decoder import Decoder
from .aotblock import AOTBlock
from torch import nn
import torch
from .base import BaseNetwork
from torch.nn.utils import spectral_norm
import os
import torch.nn.functional as F

# class Generator(BaseNetwork):
#     def __init__(self, block_number = 2):
#         super(Generator, self).__init__()
#         self.encoder = Encoder()
#         self.middleblock = nn.Sequential(*[AOTBlock(256, [1,2,4,8]) for _ in range(block_number)])
#         self.decoder = Decoder()

#         self.init_weights()
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.middleblock(x)
#         x = self.decoder(x)
#         return x
    

class InpaintGenerator(BaseNetwork):
    def __init__(self):  # 1046
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, 7),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
        )

        # Hardcoded rates
        dilation_rates = [1, 2, 4, 8]

        self.middle = nn.Sequential(*[AOTBlock(256, dilation_rates) for _ in range(8)])
        self.decoder = nn.Sequential(
            UpConv(256, 128), nn.ReLU(True), UpConv(128, 64), nn.ReLU(True), nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

        self.init_weights()

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        print("After concatenating the mask and image : ",x.shape)
        torch.cuda.empty_cache()
        print('encoder')
        with torch.no_grad():
          x = self.encoder(x)
        torch.cuda.empty_cache()
        print('AOT Block')
        x = self.middle(x)
        torch.cuda.empty_cache()
        print('decoder')
        with torch.no_grad():
          x = self.decoder(x)
        x = torch.tanh(x)
        return x




class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))
    

class pretrainedGenerator(BaseNetwork):
    def __init__(self, block_number = 2, pretrained_path = r'C:\Aakrit\College\8th Sem\Major Project\aotgan(scratch)\pretrained_models\celebahq\celebahq'):
        super(pretrainedGenerator, self).__init__()
        self.generator = InpaintGenerator()
        self.pretrained_path = pretrained_path
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.load()

        
        self.block_number = block_number

        
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1,),
            nn.Tanh()
        )

        print('Initializing random weights for decoder...')

        # self.decoder.init_weights(init_type="kaiming", gain=0.02)  # Call init_weights only for decoder

        

    
    def load(self):


        """
        Load generator, discriminator, and optimizers from saved checkpoints.
        Also loads the iteration count separately.
        """

        print('Loading the pretrained Generator...')
        try:
            # Load generator
            gpath = os.path.join(self.pretrained_path, "G.pt")
            self.generator.load_state_dict(torch.load(gpath, map_location=self.device))
            print(f"[**] Loaded generator from {gpath}")
        except Exception as e:
            print(f"[!!] Failed to load generator: {e}")

        


    def forward(self, x):

        with torch.no_grad():
            x = self.generator.encoder(x)

            for b_n in range(0, self.block_number):
                print('Entering ',b_n)
                x = self.generator.middle[b_n](x)

            x = self.generator.decoder(x)
        x = self.decoder(x)
        return x



# netG = InpaintGenerator()
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