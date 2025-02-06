import torch
from torch import nn
from .base import VGG19, gaussian_blur
import importlib
import sys
import os
import torch.nn.functional as F
# L1Loss
class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
        self.l1 = nn.L1Loss()
    
    def forward(self, x, y):
        return self.l1(x, y)


# l1loss = L1()
# x, y = torch.rand(1,3,512,512), torch.rand(1,3,512,512)

# print(l1loss(x, y))

# Perceptual Loss

class Perceptual(nn.Module):
    def __init__(self):
        super(Perceptual, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()

    
    def forward(self, x, y):
        vgg_x, vgg_y = self.vgg(x), self.vgg(y)
        loss = 0

        for i in range(1,6):
            loss += self.criterion(vgg_x[f"relu{i}_1"], vgg_y[f"relu{i}_1"])

        return loss



# x, y = torch.rand(1,3,512,512).to(torch.device('cuda')), torch.rand(1,3,512,512).to(torch.device('cuda'))
# perceptual = Perceptual().to(torch.device('cuda'))
# print(perceptual(x, y))

# Style Loss

class Style(nn.Module):
    def __init__(self):
        super(Style, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()

    def compute_gram(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * c)
        return G

    def forward(self, x, y):
        vgg_x, vgg_y = self.vgg(x), self.vgg(y)

        loss = 0.0

        prefix = [2, 3, 4, 5]
        postfix = [2, 4, 4, 2]

        for i in range(0,4):
            loss += self.criterion(self.compute_gram(vgg_x[f"relu{prefix[i]}_{postfix[i]}"]), self.compute_gram(vgg_y[f"relu{prefix[i]}_{postfix[i]}"]))

        return loss

# x, y = torch.rand(1,3,512,512).to(torch.device('cuda')), torch.rand(1,3,512,512).to(torch.device('cuda'))
# style = Style().to(torch.device('cuda'))
# print(style(x, y))

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.criterion = nn.MSELoss()
        

    def forward(self, real, fake, mask, netD):
        fake_detach = fake.detach()

        g_fake = netD(fake)
        d_fake = netD(fake_detach)
        d_real = netD(real)


        

        _, _, h, w = g_fake.size()
        b, c, ht, wt = mask.size()
        
        # Handle inconsistent size between outputs and masks
        if h != ht or w != wt:
            g_fake = F.interpolate(g_fake, size=(ht, wt), mode='bilinear', align_corners=True)
            d_fake = F.interpolate(d_fake, size=(ht, wt), mode='bilinear', align_corners=True)
            d_real = F.interpolate(d_real, size=(ht, wt), mode='bilinear', align_corners=True)

        d_fake_label = gaussian_blur(1-mask, kernel_size = (71,71), sigma = (10,10))
        d_real_label = torch.ones_like(d_real)
        g_label = torch.ones_like(g_fake)

        print('Shape of d_fake and d_fake_label : ',d_fake.shape, d_fake_label.shape)
        adv_d_loss = self.criterion(d_fake, d_fake_label) + self.criterion(d_real, d_real_label)
        adv_g_loss = self.criterion(g_fake, g_label) * mask / torch.mean(mask)

        return adv_g_loss.mean(), adv_d_loss.mean()

    
# # Add the project root directory to sys.path
# project_root = "C:/Aakrit/College/8th Sem/Major Project/aotgan(scratch)"
# sys.path.append(project_root)

# # Import the aotgan module from the models directory
# net = importlib.import_module("models.aotgan")

# netD = net.Discriminator().to(torch.device("cuda"))
# real, fake, mask = torch.rand(1,3,512,512).to(torch.device('cuda')), torch.rand(1,3,512,512).to(torch.device('cuda')), torch.rand(1,3,512,512).to(torch.device('cuda'))
# advloss = AdversarialLoss().to(torch.device('cuda'))
# print(advloss(real, fake, mask, netD))

def compute_loss(image, pred_img, comp_img, mask, netD):
    l1loss = L1().to(torch.device('cuda'))
    perceptual = Perceptual().to(torch.device('cuda'))
    style = Style().to(torch.device('cuda'))
    advloss = AdversarialLoss().to(torch.device('cuda'))


    l1 = l1loss(image, pred_img)
    per = perceptual(image, comp_img)
    sty = style(image, comp_img)
    g_adv_loss, d_adv_loss = advloss(image, comp_img, mask, netD)

    loss_dict = {
        'l1' : l1,
        'per' : per,
        'sty' : sty,
        'g_adv_loss' : g_adv_loss,
        'd_adv_loss' : d_adv_loss
    }
    

    return loss_dict
