import importlib
from torch import optim
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from customdataset import data
import os
from glob import glob
from tqdm import tqdm
import torchvision.utils as vutils
from torchvision.utils import make_grid


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class Trainer():
    def __init__(self):
        super(Trainer, self).__init__()
        net = importlib.import_module("models." + "unetGAN")
        self.netG = net.Generator()
        self.netD = net.Discriminator()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.traindl = DataLoader(data, batch_size=4, shuffle=True, pin_memory=True)
        self.traindl = sample_data(self.traindl)
        self.val_dl = DataLoader(data, batch_size=4, shuffle=True, pin_memory=True)
        self.val_dl = sample_data(self.val_dl)
        self.lr = 0.0001
        self.optG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0, 0.9))
        self.optD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0, 0.9))


        self.tensorboard_directory = r'C:\Aakrit\College\8th Sem\Major Project\aotgan(scratch)\tensorboard_logs\unetGAN'
        self.model_save_directory = r'C:\Aakrit\College\8th Sem\Major Project\aotgan(scratch)\model_directory\unetGAN'

        # Hyperparameters for printing and saving
        self.print_every = 10
        self.log_every = 10
        self.save_every = 100
        self.val_every = 100

        self.total_iterations = 10000
        self.currentiteration = 0

        self.netG, self.netD = self.netG.to(self.device), self.netD.to(self.device)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.tensorboard_directory)

        # Load models and optimizers if checkpoints exist
        self.load()

    def load(self):
        """
        Load generator, discriminator, and optimizers from saved checkpoints.
        Also loads the iteration count separately.
        """
        try:
            # Load generator
            gpath = os.path.join(self.model_save_directory, "G.pt")
            self.netG.load_state_dict(torch.load(gpath, map_location=self.device))
            print(f"[**] Loaded generator from {gpath}")
        except Exception as e:
            print(f"[!!] Failed to load generator: {e}")

        try:
            # Load discriminator
            dpath = os.path.join(self.model_save_directory, "D.pt")
            self.netD.load_state_dict(torch.load(dpath, map_location=self.device))
            print(f"[**] Loaded discriminator from {dpath}")
        except Exception as e:
            print(f"[!!] Failed to load discriminator: {e}")

        try:
            # Load optimizers
            opath = os.path.join(self.model_save_directory, "O.pt")
            checkpoint = torch.load(opath, map_location=self.device)
            self.optG.load_state_dict(checkpoint["optimG"])
            self.optD.load_state_dict(checkpoint["optimD"])
            print(f"[**] Loaded optimizers from {opath}")
        except Exception as e:
            print(f"[!!] Failed to load optimizers: {e}")

        try:
            # Load iteration count
            it_path = os.path.join(self.model_save_directory, "iteration.txt")
            with open(it_path, "r") as f:
                self.currentiteration = int(f.read().strip())
            print(f"[**] Loaded iteration count: {self.currentiteration}")
        except Exception as e:
            print(f"[!!] Failed to load iteration count, starting from 0: {e}")
            self.currentiteration = 0  # Start from zero if the file doesn't exist

    def save(self):
        """
        Save generator, discriminator, and optimizers to checkpoints.
        Save iteration count separately in a text file.
        """
        os.makedirs(self.model_save_directory, exist_ok=True)

        torch.save(self.netG.state_dict(), os.path.join(self.model_save_directory, "G.pt"))
        torch.save(self.netD.state_dict(), os.path.join(self.model_save_directory, "D.pt"))
        torch.save(
            {"optimG": self.optG.state_dict(), "optimD": self.optD.state_dict()},
            os.path.join(self.model_save_directory, "O.pt"),
        )

        # Save iteration count
        it_path = os.path.join(self.model_save_directory, "iteration.txt")
        with open(it_path, "w") as f:
            f.write(str(self.currentiteration))

        print(f"[**] Saved models, optimizers, and iteration {self.currentiteration}")

    def validate(self):
        """
        Runs validation on the dataset and logs images and losses to TensorBoard.
        """
        print("[**] Running validation...")
        lossfn = importlib.import_module("loss.loss")

        total_val_loss = 0
        num_batches = 0
        avg_loss_dict = {}

        with torch.no_grad():  # Disable gradient computation for validation
            for _ in range(10):  # Validate on 10 batches (you can modify this number)
                image, mask = next(self.val_dl)
                image, mask = image.to(self.device), mask.to(self.device)
                masked_image = (image * (1 - mask).float()) + mask
                x = torch.cat([masked_image, mask], dim=1)
                pred_img = self.netG(x)
                comp_img = (1 - mask) * image + mask * pred_img

                all_loss_dict = lossfn.compute_loss(image, pred_img, comp_img, mask, self.netD)
                val_loss = (
                    all_loss_dict['l1'] +
                    0.1 * all_loss_dict['per'] +
                    250 * all_loss_dict['sty'] +
                    0.01 * all_loss_dict['g_adv_loss']
                )
                total_val_loss += val_loss.item()
                num_batches += 1

                for key, val in all_loss_dict.items():
                    if key not in avg_loss_dict:
                        avg_loss_dict[key] = 0
                    avg_loss_dict[key] += val.item()

            avg_val_loss = total_val_loss / num_batches

            for key in avg_loss_dict:
                avg_loss_dict[key] /= num_batches
                self.writer.add_scalar(f"Validation/{key}", avg_loss_dict[key], self.currentiteration)

            # Log validation loss
            self.writer.add_scalar("Validation Loss", avg_val_loss, self.currentiteration)

            # Log images to TensorBoard
            self.writer.add_image("Validation Mask", make_grid(mask), self.currentiteration)
            self.writer.add_image("Validation Original", make_grid((image + 1.0) / 2.0), self.currentiteration)
            self.writer.add_image("Validation Predicted", make_grid((pred_img + 1.0) / 2.0), self.currentiteration)
            self.writer.add_image("Validation Composite", make_grid((comp_img + 1.0) / 2.0), self.currentiteration)

        print(f"[**] Validation completed. Avg Loss: {avg_val_loss:.4f}")


    def train(self):
        """
        Training loop where each batch is treated as an iteration.
        Logs images and losses to TensorBoard without saving them locally.
        """
        # Import the loss module
        lossfn = importlib.import_module("loss.loss")

        # Initialize tqdm progress bar
        pbar = range(self.currentiteration, self.total_iterations)


        for idx in pbar:
            self.currentiteration += 1
            print('Iteration : ', self.currentiteration, '/', self.total_iterations)
            image, mask = next(self.traindl)

            # Move data to the appropriate device
            image, mask = image.to(self.device), mask.to(self.device)

            # Create masked image
            masked_image = (image * (1 - mask).float()) + mask

            # Concatenate masked image and mask
            x = torch.cat([masked_image, mask], dim=1)

            # Generate predicted image
            pred_img = self.netG(x)

            # Compute composite image
            comp_img = (1 - mask) * image + mask * pred_img

            # Compute losses
            all_loss_dict = lossfn.compute_loss(image, pred_img, comp_img, mask, self.netD)

            # Combine losses
            all_loss_dict['total_g_loss'] = (
                all_loss_dict['l1'] +
                0.1 * all_loss_dict['per'] +
                250 * all_loss_dict['sty'] +
                0.01 * all_loss_dict['g_adv_loss']
            )
            all_loss_dict['total_d_loss'] = all_loss_dict['d_adv_loss']

            # Sum generator and discriminator losses
            gLoss = all_loss_dict['total_g_loss']
            dLoss = all_loss_dict['total_d_loss']

            # Log losses to TensorBoard
            if self.currentiteration % self.log_every == 0 or self.currentiteration == 1:
                for key, val in all_loss_dict.items():
                    self.writer.add_scalar(key, val.item(), self.currentiteration)

                # Log images separately in TensorBoard
                self.writer.add_image("mask", make_grid(mask), self.currentiteration)
                self.writer.add_image("orig", make_grid((image + 1.0) / 2.0), self.currentiteration)
                self.writer.add_image("pred", make_grid((pred_img + 1.0) / 2.0), self.currentiteration)
                self.writer.add_image("comp", make_grid((comp_img + 1.0) / 2.0), self.currentiteration)


            # Print losses every `print_every` iterations
            # if self.currentiteration % self.print_every == 0:
            if self.currentiteration % self.print_every == 0:
                print(
                    f"Iter: {self.currentiteration}, G Loss: {all_loss_dict['total_g_loss'].item():.4f}, "
                    f"D Loss: {all_loss_dict['total_d_loss'].item():.4f}"
                )
            



            # Backpropagation
            self.optG.zero_grad()
            self.optD.zero_grad()
            gLoss.backward()  # Backpropagate combined loss
            dLoss.backward()
            self.optG.step()
            self.optD.step()

            if self.currentiteration % self.val_every == 0:
                self.validate()

            # Save models every `save_every` iterations
            if self.currentiteration % self.save_every == 0:
                self.save()

            

        # Close TensorBoard writer
        self.writer.close()
        print("[**] Training complete!")



if __name__ == '__main__':
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    trainer = Trainer()
    trainer.train()