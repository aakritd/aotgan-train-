import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        
        # 1x1 convolutions to generate Query, Key, Value
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling factor

    def forward(self, x):
        batch, C, H, W = x.shape

        # Compute Query, Key, Value
        Q = self.query(x).view(batch, C // 8, -1)  # (B, C//8, H*W)
        K = self.key(x).view(batch, C // 8, -1)    # (B, C//8, H*W)
        V = self.value(x).view(batch, C, -1)       # (B, C, H*W)

        # Compute Attention Scores (dot product of Q and K)
        attention = torch.bmm(Q.permute(0, 2, 1), K)  # (B, H*W, H*W)
        attention = F.softmax(attention, dim=-1)  # Apply softmax

        # Apply Attention to Value (V)
        out = torch.bmm(attention, V.permute(0, 2, 1))  # (B, H*W, C)
        out = out.permute(0, 2, 1).view(batch, C, H, W)  # Reshape back to original shape

        # Residual Connection
        out = self.gamma * out + x
        return out


class InpaintingGenerator(nn.Module):
    def __init__(self):
        super(InpaintingGenerator, self).__init__()

        # Encoder (ResNet-based)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[1:-2])  # Remove last layers

        # Attention Block
        self.attention = SelfAttention(in_channels=512)

        # Decoder (Upsampling)
        self.decoder = nn.Sequential(
            # ConvTranspose2d: Increase the spatial resolution
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, output_padding=0), # 256x256 -> 512x512
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),  # 64x64 -> 128x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, output_padding=0),  # 64x64 -> 128x128
            nn.Tanh()  # Output scaled to [-1, 1]
        )

    def forward(self, x):
        x = self.conv1(x)  # 1, 4, 512, 512 -> 1, 64, 256, 256
        x = self.encoder(x)  # Encoding part (downsampling)
        x = self.attention(x)  # Apply attention mechanism    
        x = self.decoder(x)  # Decoding part (upsampling)
        return x


# Test the Generator with the new decoder
netG = InpaintingGenerator()
x = torch.rand(1, 4, 512, 512)  # Input with batch size 1 and 4 channels
output = netG(x)
print(output.shape)  # Expected output: torch.Size([1, 3, 512, 512])
