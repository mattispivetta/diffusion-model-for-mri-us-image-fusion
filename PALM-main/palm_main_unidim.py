#**************************************************************************
# Denoising Diffusion Model for Multi-modality Image Fusion with Proximal
# Alternating Linearized Minimization algorithm
# Author: Tom Longin (2026 June)
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
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma
from skimage.restoration import denoise_nl_means
import bm3d

# --- Fichiers ---
from matlab_tools import load_dncnn
from palm_utils_unidim import estimate_c, FusionPALM


def show_image(img, title='Image'):
    """Affiche une image grayscale"""
    plt.figure(figsize=(10,5))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def chooseDataset_unidim():
    np.set_printoptions(precision=16)
    # --- Chargement des images ---
    irm_data = scipy.io.loadmat('images/Data3/irm.mat')
    us_data = scipy.io.loadmat('images/Data3/us.mat')
    irm = irm_data['irm'].astype(np.float64) #irm1_gt pour Data2_GT, irm1 pour Data2, irm_gt pour Data3_GT, irm pour Data3
    us = us_data['us'].astype(np.float64) #us1_gt pour Data2_GT, us1 pour Data2, us_gt pour Data3_GT, us pour Data3
    show_image(irm, title='IRM')
    show_image(us, title='US')
    return irm, us


def solve_PALM_unidim(irm, us):
    # --- Coefficients ---
    c = estimate_c(irm, us)

    # --- Normalisation ---
    ym = irm.astype(np.float64) / np.max(irm)
    yu = us.astype(np.float64) / np.max(us)
    
    # --- Initialisation --- 
    d = 1 # uniquement pour Data2 et Data3

    # --- Débruitage US ---
    ## -- dncnn --
    xu0 = load_dncnn(yu)
    
    ## -- bm3d --
    #xu0 = bm3d.bm3d(yu, sigma_psd=estimate_sigma(yu, channel_axis=None, average_sigmas=True))
    
    ## -- nlm --
    #xu0 = denoise_nl_means(yu, h=1.15 * estimate_sigma(yu, channel_axis=None, average_sigmas=True), fast_mode=False, patch_size=5, patch_distance=6, channel_axis=None )
    
    show_image(xu0, title='US débruité')
    #xu0 = yu
    
    # --- Paramètres de régularisation ---
    # Data2: alpha = 0.2
    # tau1 = 1e-12        
    # tau2 = 3e-6
    # tau3 = 2e-6
    # tau4 = 1e-5

    # Data3: alpha = 0.2
    tau1 = 1e-5
    tau2 = 5e-1
    tau3 = 2e-2
    tau4 = 1e-5

    # --- Nombre d'itérations ---
    m_iteration = 10
    
    # --- PALM ---
    x2, progress = FusionPALM(ym, xu0, c, tau1, tau2, tau3, tau4, d, m_iteration)
    return x2, progress

irm, us = chooseDataset_unidim()
x2, progress = solve_PALM_unidim(irm, us)
show_image(x2, 'Fusion')