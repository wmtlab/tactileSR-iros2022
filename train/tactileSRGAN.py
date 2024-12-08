import os
import sys
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import skimage.metrics as metrics
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.net import SRCNN_MSRN, Dis_Net, compute_gradient_penalty
from utility.tactileDataLoader import TactileDataLoader
from config import *

# Initialize SummaryWriter
writer = SummaryWriter(root_log_srgan)

# Device setup
print(f"{TRAIN_NAME_SRGAN} | CUDA device: {CUDA_DEVICE_NUM}")

# ----------------------- Model and Loss -----------------------
GNet = SRCNN_MSRN(feature_layers_num=LAY_NUM, is_init=False).to(device)
DNet = Dis_Net(LRB_layer_num=LAY_NUM//2, is_init=False).to(device)
criterion_mse = nn.MSELoss().to(device)

# Optimizers
optimizerG = optim.Adam(GNet.parameters(), lr=GAN_G_LR, weight_decay=WEIGHT_DECAY)
optimizerD = optim.Adam(DNet.parameters(), lr=GAN_D_LR, weight_decay=WEIGHT_DECAY)

# ----------------------- Data Preparation -----------------------
dataset_file = os.path.join(root_path, 'dataset', f'TSR_data_x{RESIZE_FACTOR}', 'train/')
tactile_dataset = TactileDataLoader(dataset_file)

# Split dataset
dataset_len = len(tactile_dataset)
train_size = int(dataset_len * 0.8)
test_size = dataset_len - train_size
train_dataset, test_dataset = random_split(tactile_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

print(f"Training batches: {len(train_loader)}")

# ----------------------- Training Functions -----------------------
def train_epoch(epoch, GNet, DNet, train_loader, optimizerG, optimizerD):
    GNet.train()
    DNet.train()
    epoch_loss = 0

    for lr_data, hr_data in train_loader:
        if IS_NORMALIZED:
            lr_data /= SCALE_VALUE
            hr_data /= SCALE_VALUE

        hr_data_z = hr_data[:, 2:3, :, :]
        lr_data = lr_data.to(device).float()
        real_data = hr_data_z.to(device).float()
        fake_data = GNet(lr_data)

        # Update Discriminator
        for _ in range(2):
            DNet.zero_grad()
            logits_real = DNet(real_data).mean()
            logits_fake = DNet(fake_data).mean()
            gradient_penalty = compute_gradient_penalty(DNet, real_data, fake_data)
            d_loss = logits_fake - logits_real + 10 * gradient_penalty
            d_loss.backward(retain_graph=True)
            optimizerD.step()

        # Update Generator
        GNet.zero_grad()
        fake_data = GNet(lr_data)
        data_loss = criterion_mse(fake_data, real_data)
        adversarial_loss = -DNet(fake_data).mean()
        g_loss = data_loss + 1e-3 * adversarial_loss
        g_loss.backward()
        optimizerG.step()

        epoch_loss += data_loss.item()

    writer.add_scalar("train_loss", epoch_loss, epoch)
    print(f"Epoch [{epoch}]: Generator Loss: {epoch_loss:.4f}",  end= "  |   ")

def evaluate(epoch, GNet, test_loader):
    GNet.eval()
    test_loss = 0
    avg_psnr = 0
    avg_ssim = 0

    with torch.no_grad():
        for lr_data, hr_data in test_loader:
            if IS_NORMALIZED:
                lr_data /= SCALE_VALUE
                hr_data /= SCALE_VALUE

            hr_data_z = hr_data[:, 2:3, :, :]
            lr_data = lr_data.to(device).float()
            hr_data_z = hr_data_z.to(device).float()

            model_out = GNet(lr_data)
            loss = criterion_mse(model_out, hr_data_z)
            test_loss += loss.item()

            model_out_np = model_out.cpu().numpy()
            hr_data_z_np = hr_data_z.cpu().numpy()

            for i in range(model_out_np.shape[0]):
                single_out = model_out_np[i, 0, :, :]
                single_hr = hr_data_z_np[i, 0, :, :]
                psnr = metrics.peak_signal_noise_ratio(single_out, single_hr, data_range=single_hr.max())
                ssim = metrics.structural_similarity(single_out, single_hr, data_range=single_hr.max())
                avg_psnr += psnr
                avg_ssim += ssim

    avg_psnr /= len(test_loader.dataset)
    avg_ssim /= len(test_loader.dataset)
    print(f"Epoch [{epoch}] | Test Loss: {test_loss:.4f} | PSNR: {avg_psnr:.4f} dB | SSIM: {avg_ssim:.4f}")
    writer.add_scalar("test_loss", test_loss, epoch)
    writer.add_scalar("test_psnr", avg_psnr, epoch)
    writer.add_scalar("test_ssim", avg_ssim, epoch)

# ----------------------- Main Training Loop -----------------------
for epoch in range(1, EPOCHS + 1):
    train_epoch(epoch, GNet, DNet, train_loader, optimizerG, optimizerD)
    evaluate(epoch, GNet, test_loader)

    if epoch > 50 and epoch % 10 == 0:
        torch.save(GNet.state_dict(), os.path.join(root_pth_srgan, f'srgan_epoch_{epoch}.pth'))
