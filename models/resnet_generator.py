import torch
import torch.nn as nn
import torchvision.models as models

class ResNetUNetGenerator(nn.Module):
    def __init__(self):
        super(ResNetUNetGenerator, self).__init__()

        #Encoder
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, stride=2, padding=0, bias=False)

        resnet = models.resnet18(pretrained = True)

      
        layers = list(resnet.children())[1:-2]
        

        self.encoder = nn.Sequential(self.conv1, *layers)

        
        # print(list(resnet.children())[1])

        # Decoder (Adjusted to get 512x512 output)
        self.decoder = nn.Sequential(
            # Upsampling layers
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=0),  # [1, 256, 256, 256]
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),  # [1, 128, 512, 512]
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),  # [1, 32, 512, 512]
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, output_padding=0),  # [1, 3, 512, 512]
            nn.Tanh()  # Output scaled to [-1, 1] range (if that's your desired output)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return (x)

x = torch.rand(1,4,512,512)
generator = ResNetUNetGenerator()
print(generator(x).shape)
