#**************************************************************************
# Denoising Diffusion Model for Multi-modality Image Fusion with Proximal
# Alternating Linearized Minimization algorithm
# Author: Mattis Pivetta (2026 August)
# University of Toulouse, IRIT
# Email: mattis.pivetta@irit.fr
#
# Copyright (2026): Mattis Pivetta
# 
# Permission to use, copy, modify, and distribute this software for
# any purpose without fee is hereby granted, provided that this entire
# notice is included in all copies of any software which is or includes
# a copy or modification of this software and in all copies of the
# supporting documentation for such software.
# This software is being provided "as is", without any express or
# implied warranty.  In particular, the authors do not make any
# representation or warranty of any kind concerning the merchantability
# of this software or its fitness for any particular purpose."
#**************************************************************************

# --- Librairies ---
from functools import partial
import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from skimage.io import imsave, imread
import warnings
import bm3d
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# --- Fichiers ---
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from guided_diffusion.PALM_DDFM_unidim import init_PALM
from util.logger import get_logger
from ResizeRight.resize_right import resize
from ResizeRight.interp_methods import cubic
import scipy

# --- Fonctions ---
def image_read(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def erase_empty(img):
    _, thresh = cv2.threshold(img, 0.99, 1, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y+h, x:x+w]
    return cropped

def augment_size(img, target_height, target_width):
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

def down_size(image, target_height, target_width):
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

if __name__ == '__main__':
    
    # --- Récupération des arguments dans un Parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str,default = 'configs/model_config_imagenet.yaml')
    parser.add_argument('--diffusion_config', type=str,default='configs/diffusion_config.yaml')                     
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./outputTLSE')
    args = parser.parse_args()
    
    # --- Logger ---
    logger = get_logger()
    
    # --- Device CPU/GPU ---
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # --- Configurations du modèle
    model_config = load_yaml(args.model_config)  
    diffusion_config = load_yaml(args.diffusion_config)
    
    # --- Chargement du modèle  ---
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # --- Chargement de la diffusion ---
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model)
    
    # --- Dossiers d'enregistrement ---
    test_folder=r"input-TLSE"     
    out_path = args.save_dir
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['recon', 'progress']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
    i = 0
    folder_list = sorted(os.listdir(test_folder))
    folder_name = 'Data3'
#############################################################################################################################
############### SI DATA .png ############################## 
    # mri_name = os.path.join(test_folder, folder_name, "Data2_irm_gt.png")
    # mri_img = image_read(mri_name, mode='GRAY')
    # us_name = os.path.join(test_folder, folder_name, "Data2_us_gt.png")
    # us_img = image_read(us_name, mode='GRAY')
##############################################################################################################################
############### SI DATA .mat ##############################
    np.set_printoptions(precision=16)
    # --- Chargement des images ---
    irm_data = scipy.io.loadmat('input-TLSE/Data3/irm.mat')
    us_data = scipy.io.loadmat('input-TLSE/Data3/us.mat')
    mri_img = irm_data['irm'].astype(np.float64) #irm1_gt pour Data2_GT, irm1 pour Data2, irm_gt pour Data3_GT, irm pour Data3
    us_img = us_data['us'].astype(np.float64) #us1_gt pour Data2_GT, us1 pour Data2, us_gt pour Data3_GT, us pour Data3
#############################################################################################################################
#############################################################################################################################
    H_us, W_us = us_img.shape
     
    # --- Vérification et resize éventuel pour respecter un multiple de 32 ---
    H_opt = ( H_us // 32) * 32
    W_opt = ( W_us // 32) * 32
    
    us_img = resize(us_img, out_shape=(H_opt,W_opt), interp_method=cubic)
    mri_img = resize(mri_img, out_shape=(H_opt,W_opt), interp_method=cubic)
    
    print(mri_img.shape, us_img.shape, us_img.shape[-2]/32, mri_img.shape[-1]/32)
#############################################################################################################################
#############################################################################################################################
    # H_us, W_us = us_img.shape
     
    # # --- Vérification et zero-padding éventuel pour respecter un multiple de 32 ---
    # H_opt = (32 - H_us % 32) % 32
    # W_opt = (32 - W_us % 32) % 32
    # us_img = np.pad(us_img, ((H_opt//2, H_opt-(H_opt//2)),(W_opt//2, W_opt-(W_opt//2))), mode="constant", constant_values=0)
    # mri_img= np.pad(mri_img, ((H_opt//2, H_opt-(H_opt//2)),(W_opt//2, W_opt-(W_opt//2))), mode="constant", constant_values=0)
    
    # print(mri_img.shape, us_img.shape, us_img.shape[-2]/32, mri_img.shape[-1]/32)
#############################################################################################################################
    palm_parameters = init_PALM(mri_img, us_img)

    # --- Conversion des images en tenseurs Torch (shape: [1, 1, H, W]) ---
    if isinstance(mri_img, np.ndarray):
        mri_tensor = torch.from_numpy(mri_img).float().to(device)
        if mri_tensor.ndim == 2:
            mri_tensor = mri_tensor.unsqueeze(0).unsqueeze(0)     # [1, 1, H, W]
    if isinstance(us_img, np.ndarray):
        us_tensor = torch.from_numpy(us_img).float().to(device)
        if us_tensor.ndim == 2:
            us_tensor = us_tensor.unsqueeze(0).unsqueeze(0)       # [1, 1, H, W]
        
    # --- Redimensionner l’image IRM à la taille de l’US ---
    H_us, W_us = us_tensor.shape[-2:]
    mri_tensor = F.interpolate(mri_tensor, size=(H_us, W_us), mode='bicubic', align_corners=False)
   
    # --- Normalisation entre [-1, 1] si les images étaient dans [0, 1] ---
    mri_tensor = (mri_tensor * 2 - 1).to(device)
    us_tensor  = (us_tensor  * 2 - 1).to(device)
        
    # --- Remplacer les anciens noms pour cohérence avec le reste du pipeline ---
    mri_img = mri_tensor
    us_img  = us_tensor
        
    # --- Vérification finale ---
    assert mri_img.shape == us_img.shape

    # --- Lancement de la fusion ---
    logger.info(f"Inference starting for image {i}.")
    seed = 3407
    torch.manual_seed(seed)
    x_start = torch.randn((mri_img.repeat(1, 3, 1, 1)).shape, device=device)  
    with torch.no_grad():
        sample = sample_fn(
            x_start=x_start,
            record=True,
            I=mri_img,
            V=us_img,
            save_root=out_path,
            img_index=folder_name,
            lamb=0.3,
            rho=0.001,
            palm_init=palm_parameters
        )
    recon_path = os.path.join(out_path, "recon")

    # --- Reconstruction ---
    sample_np = sample.detach().cpu().squeeze().numpy()                  # [3, H, W]
    sample_np = np.transpose(sample_np, (1, 2, 0))                       # [H, W, 3]
    sample_np_y = cv2.cvtColor(sample_np, cv2.COLOR_RGB2YCrCb)[:, :, 0]  # extraction canal Y
    sample_np_y = (sample_np_y - sample_np_y.min()) / (sample_np_y.max() - sample_np_y.min())
    sample_np_y = (sample_np_y * 255).astype(np.uint8)
    # --- Sauvegarde ---
    imsave(os.path.join(out_path, 'recon', f"{folder_name}_palm_fused.png"), sample_np_y)
    i += 1
