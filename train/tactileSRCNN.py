#!/usr/bin/python3
import os
import sys
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import skimage.metrics as metrics
from math import log10

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.net import SRCNN_MSRN
from utility.tactileDataLoader import TactileDataLoader
from train.config import *


# Initialize SummaryWriter
writer = SummaryWriter(root_log_srcnn)

# Device setup
print(f"{TRAIN_NAME_SRCNN} | CUDA device: {CUDA_DEVICE_NUM}")

# ----------------------- Model and Loss -----------------------
model = SRCNN_MSRN(feature_layers_num=LAY_NUM, is_init=True).to(device)
criterion = nn.MSELoss().to(device)

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

# ----------------------- Training Loop -----------------------
def train_epoch(epoch, model, optimizer, train_loader):
    model.train()
    epoch_loss = 0

    for lr_data, hr_data in train_loader:
        if IS_NORMALIZED:
            lr_data /= SCALE_VALUE
            hr_data /= SCALE_VALUE

        hr_data_z = hr_data[:, 2:3, :, :]

        lr_data = lr_data.to(device).float()
        hr_data_z = hr_data_z.to(device).float()

        optimizer.zero_grad()
        model_out = model(lr_data)
        loss = criterion(model_out, hr_data_z)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"TRAIN | Epoch [{epoch}]: Loss: {epoch_loss:.4f} lr_rate: {optimizer.param_groups[0]['lr']:.6f}", end='   |  ')
    writer.add_scalar("train_loss", epoch_loss, epoch)

def evaluate(epoch, model, test_loader):
    model.eval()
    test_loss = 0
    total_psnr = 0
    total_ssim = 0

    with torch.no_grad():
        for lr_data, hr_data in test_loader:
            if IS_NORMALIZED:
                lr_data /= SCALE_VALUE
                hr_data /= SCALE_VALUE

            hr_data_z = hr_data[:, 2:3, :, :]

            lr_data = lr_data.to(device).float()
            hr_data_z = hr_data_z.to(device).float()

            model_out = model(lr_data)
            loss = criterion(model_out, hr_data_z)
            test_loss += loss.item()

            model_out_np = model_out.cpu().numpy()
            hr_data_z_np = hr_data_z.cpu().numpy()

            for i in range(model_out_np.shape[0]):
                single_out = model_out_np[i, 0, :, :]
                single_hr = hr_data_z_np[i, 0, :, :]
                psnr = metrics.peak_signal_noise_ratio(single_out, single_hr, data_range=single_hr.max())
                ssim = metrics.structural_similarity(single_out, single_hr, data_range=single_hr.max())
                total_psnr += psnr
                total_ssim += ssim

    avg_psnr = total_psnr / len(test_loader.dataset)
    avg_ssim = total_ssim / len(test_loader.dataset)
    print(f"EVAL | Test Loss: {test_loss:.4f} | PSNR: {avg_psnr:.4f} dB | SSIM: {avg_ssim:.4f}")
    writer.add_scalar("test_loss", test_loss, epoch)
    writer.add_scalar("test_psnr", avg_psnr, epoch)
    writer.add_scalar("test_ssim", avg_ssim, epoch)

# ----------------------- Main Loop -----------------------
optimizer = optim.Adam(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)  # Adjust step_size and gamma as needed

for epoch in range(1, EPOCHS + 1):
    train_epoch(epoch, model, optimizer, train_loader)
    evaluate(epoch, model, test_loader)

    scheduler.step()

    if epoch > 90 and epoch % 50 == 0:
        model_save_path = os.path.join(root_pth_srcnn, f'srcnn_epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_save_path)
