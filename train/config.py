import os
import torch
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

# ----------------------- General Configuration -----------------------
LAY_NUM = 6
CUDA_DEVICE_NUM = 0
RESIZE_FACTOR = 10
BATCH_SIZE = 32
TEST_BATCH_SIZE = 8
EPOCHS = 300
WEIGHT_DECAY = 1e-4
INIT_LR = 1e-4     # for SRCNN
GAN_G_LR = 1e-4    # for SRGAN
GAN_D_LR = 1e-4    # for SRGAN
IS_NORMALIZED = False
SCALE_VALUE = 1000
SCALE_VALUE_Z = 1000
SCALE_VALUE_XY = 1000
TRAIN_NAME_SRCNN = f"srcnn_data_x{RESIZE_FACTOR}"
TRAIN_NAME_SRGAN = f"srgan_data_x{RESIZE_FACTOR}"

# Paths
dirname, _ = os.path.split(os.path.abspath(__file__))
root_path = os.path.dirname(dirname) + '/'
root_log_srcnn = os.path.join(root_path, 'logs', 'srcnn', TRAIN_NAME_SRCNN)
root_pth_srcnn = os.path.join(root_path, 'pth', 'srcnn_pth', TRAIN_NAME_SRCNN)
root_log_srgan = os.path.join(root_path, 'logs', 'srgan', TRAIN_NAME_SRGAN)
root_pth_srgan = os.path.join(root_path, 'pth', 'srgan_pth', TRAIN_NAME_SRGAN)

# Ensure directories exist
os.makedirs(root_pth_srcnn, exist_ok=True)
os.makedirs(root_pth_srgan, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(CUDA_DEVICE_NUM)

