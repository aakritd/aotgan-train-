import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(

            # Layer 1
            nn.ReflectionPad2d(3),
            nn.Conv2d(4,64,7),
            nn.ReLU(),

            # Layer 2
            nn.Conv2d(64,128,4,2,1),
            nn.ReLU(),

            # Layer 3
            nn.Conv2d(128,256,4,2,1),
            nn.ReLU()
        )
    
    def forward(self,x):
        output = self.encoder(x)
        return output

# encoder = Encoder()
# x = torch.rand(1,4,512,512)
# print(encoder(x).shape)


'''
1) Why reflection padding is used in the beginning?
2) Why not use normal padding of the convolution?

'''