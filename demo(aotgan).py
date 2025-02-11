import torch
import importlib
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
net = importlib.import_module("models." + "aotgan")
netG = net.Generator(2)


gpath = os.path.join(r'C:\Aakrit\College\8th Sem\Major Project\aotgan(scratch)\model_directory\aotgan(scratch)', "G.pt")
# gpath = os.path.join(r'C:\Aakrit\College\8th Sem\Major Project\aotgan(scratch)\kaggle_output\1\model_dir', "G.pt")
netG.load_state_dict(torch.load(gpath, map_location=torch.device('cuda')))
print(f"[**] Loaded generator from {gpath}")


image_path = r'C:\Aakrit\College\8th Sem\Major Project\Image Inpainting\deepfill(scratch)\image.png'
mask_path = r'C:\Aakrit\College\8th Sem\Major Project\aotgan(scratch)\mask2.png'

image_transformation = transforms.Compose(
    [
        transforms.Resize((512,512)),
        transforms.ToTensor()
    ]
    
)
mask_transformation = transforms.ToTensor()



img = Image.open(image_path)
mask = Image.open(mask_path)


mask = mask_transformation(mask.convert('L'))
img = image_transformation(img.convert('RGB'))* 2.0 - 1.0

print(mask.shape, img.shape)

mask = mask.unsqueeze(0)
img = img.unsqueeze(0)

masked_img = (img * (1 - mask).float()) + mask

inp = torch.cat([masked_img, mask], dim=1)

output = netG(inp)

# Convert output to numpy for visualization
output = output.squeeze(0).detach().cpu()  # Remove batch dim
output = (output + 1) / 2  # Normalize from [-1,1] to [0,1]
output = output.permute(1, 2, 0) # Convert from (C, H, W) to (H, W, C)

# Display image
plt.imshow(output)
plt.axis("off")  # Hide axes
plt.show()

print(output.shape)


