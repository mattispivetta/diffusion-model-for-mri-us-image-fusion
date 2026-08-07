# --- Librairies ---
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d
from skimage.restoration import estimate_sigma
from skimage.restoration import denoise_nl_means
import bm3d

# --- Fichiers ---
from .utils_palm2 import estimate_c, FusionPALM
from .matlab_tools import load_dncnn
from ResizeRight.resize_right import resize
from ResizeRight.interp_methods import cubic

print("Utilisation de Palm_unidim")

def init_PALM(irm, us):
    #Initialisation des paramètres et des variables de PALM
    # Force en array 2D float64 (au cas où ce sont des tensors ou 3D)
    irm = np.array(irm.squeeze(), dtype=np.float64)
    us = np.array(us.squeeze(), dtype=np.float64)

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
    #xu0 = denoise_nl_means(yu, h=1.15 * estimate_sigma(yu, channel_axis=None, average_sigmas=True), fast_mode=False, patch_size=5, patch_distance=6, channel_axis=None)
    #xu0 = yu
    
    # --- Paramètres de régularisation ---
    # Data2 : alpha = 0.2             # Data2 test2 : alpha = 0.2            # Data2 test3 : alpha = 0.2        # Data2 palm : alpha = 0.2
    # tau1 = 1e-12                    # tau1 = 1e-4                          # tau1 = 1e-2                      # tau1 = 1e-12
    # tau2 = 3e-6                     # tau2 = 3e-8                          # tau2 = 1e-5                      # tau2 = 3e-6 
    # tau3 = 2e-2                     # tau3 = 5e-2                          # tau3 = 5e-3                      # tau3 = 2e-6
    # tau4 = 1e-4                     # tau4 = 2e-3                          # tau4 = 3e-12                     # tau4 = 1e-5

    # Data3 : alpha = 0.5             # Data3 test2 : alpha = 0.5            # Data3 test3 : alpha = 0.5        # Data3 palm : alpha = 0.2 
    # tau1 = 1e-7                     # tau1 = 1e-5                          # tau1 = 1e-12                     # tau1 = 1e-5
    # tau2 = 3e-6                     # tau2 = 3e-5                          # tau2 = 8e-4                      # tau2 = 5e-1
    # tau3 = 2e-2                     # tau3 = 2e-8                          # tau3 = 3e-2                      # tau3 = 2e-2
    # tau4 = 1e-4                     # tau4 = 1e-2                          # tau4 = 1e-7                      # tau4 = 1e-5

    tau1 = 1e-5
    tau2 = 5e-1
    tau3 = 2e-2  
    tau4 = 1e-5 
    
    # --- Nombre d'itérations ---
    m_iteration = 2
    
    # Paramètres PALM
    return {
        "ym": ym, "xu0": xu0, "c": c,
        "tau1": tau1, "tau2": tau2, "tau3": tau3, "tau4": tau4,
        "d": d, "m_iteration": m_iteration
    }


def fusion_onestep(f_pre, palm_init=None):
    """
    Args:
        f_pre     : prédiction courante du modèle de diffusion [B, 1, H, W]
        palm_init : dictionnaire contenant les paramètres PALM pré-initialisés
        plot      : bool, affiche l'image fusionnée à t=0
    Returns:
        x_fused   : image fusionnée [B, 1, H, W]
    """
    assert palm_init is not None, "palm_init est requis"

    device = f_pre.device
    batch_size = f_pre.shape[0]

    # Extraction des paramètres depuis palm_init
    ym = palm_init["ym"]       # IRM normalisée
    xu0 = palm_init["xu0"]     # US débruitée
    c = palm_init["c"]
    tau1 = palm_init["tau1"]
    tau2 = palm_init["tau2"]
    tau3 = palm_init["tau3"]
    tau4 = palm_init["tau4"]
    d = palm_init["d"]         # Coefficient de super-résolution
    m_iteration = palm_init["m_iteration"]

    # Exécution de PALM (1 seule image, donc pas vectorisée batch)
    f_np = f_pre[0, 0].detach().cpu().numpy()
    x2 = FusionPALM(ym, xu0, c, tau1, tau2, tau3, tau4, d, m_iteration, f_np)
    #plt.imsave("result.png", x2, cmap="gray")
    # Conversion en tenseur PyTorch [B, 1, H, W]
    x2_tensor = torch.from_numpy(x2).float().unsqueeze(0).unsqueeze(0).to(device)
    if batch_size > 1:
        x2_tensor = x2_tensor.repeat(batch_size, 1, 1, 1)
    return x2_tensor