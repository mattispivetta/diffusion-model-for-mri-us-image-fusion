import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import sobel
from skimage import io, img_as_ubyte
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
from sklearn.metrics import mutual_info_score
from sewar.full_ref import vifp


def compute_entropy(image):
    """
    Entropy (EN) measures the amount of information contained in a fused image on the
    basis of information theory.
    The larger the EN, the more information is contained
    in the fused image and the better the performance of the fusion method. However, EN may
    be influenced by noise; the more noise the fused image contains, the larger the EN. Therefore,
    EN is usually used as an auxiliary metric.
    """
    return shannon_entropy(image)

def compute_sd(image):
    """
    The standard deviation (SD) metric is based on the statistical concept that reflects the
    distribution and contrast of the fused image.
    Regions with high contrast always attract human attention due to the sensitivity of the human
    visual system to contrast.
    Therefore, a fused image with high contrast often results in a large SD, which means that
    the fused image achieves a good visual effect.
    """
    return np.std(image)

def compute_mi(image_us, image_us_debruite, image_irm, img_fused):
    """
    The mutual information (MI) metric is a quality index that measures the amount of
    information that is transferred from source images to the fused image.
    A large MI metric means that considerable information is transferred from source images
    to the fused image, which indicates a good fusion performance.
    """
    hist_us, _, _ = np.histogram2d(image_us.ravel(), img_fused.ravel(), bins=20)
    hist_us_debruite, _, _ = np.histogram2d(image_us_debruite.ravel(), img_fused.ravel(), bins=20)
    hist_irm, _, _ = np.histogram2d(image_irm.ravel(), img_fused.ravel(), bins=20)
    return [mutual_info_score(None, None, contingency=hist_us), mutual_info_score(None, None, contingency=hist_irm), mutual_info_score(None, None, contingency=hist_us_debruite)]

def compute_vif(image_us, image_us_debruite, image_irm, fused_image):
    """
    This is a VIF Pixel Based.
    The visual information fidelity (VIF) metric measures the information fidelity of the
    fused image [298], which is consistent with the human visual system.
    """
    return [vifp(image_us, fused_image), vifp(image_irm, fused_image), vifp(image_us_debruite, fused_image)]

def compute_edge_metric(image_us, image_us_debruite, image_irm, fused_image):
    """
    QAB/F measures the amount of edge information that is transferred from source images
    to the fused image and is based on the assumption that the edge information in the source
    images is preserved in the fused image.
    A large QAB/F means that considerable edge information is transferred to the fused image.
    """
    return [np.corrcoef(sobel(fused_image).flatten(), 
                       sobel(image_us).flatten())[0, 1], np.corrcoef(sobel(fused_image).flatten(), 
                                                                      sobel(image_irm).flatten())[0, 1], np.corrcoef(sobel(fused_image).flatten(),
                                                                                                                     sobel(image_us_debruite).flatten())[0, 1]]

def compute_ssim(image_us, image_us_debruite, image_irm, fused_image):
    return [ssim(image_us, fused_image), ssim(image_irm, fused_image), ssim(image_us_debruite, fused_image)]


def compute_mse(image_gt, image_fused):
    image_gt = image_gt.astype(np.float64)   ;   image_fused = image_fused.astype(np.float64)
    eqm = np.sum((image_gt - image_fused) ** 2) / image_gt.size
    return eqm


def compute_psnr(image_gt, image_fused):
    image_gt = image_gt.astype(np.float64)   ;   image_fused = image_fused.astype(np.float64)
    d = max(np.max(image_gt), np.max(image_fused))
    psnr = 10 * np.log10(d**2 / compute_mse(image_gt, image_fused))
    return psnr


def compute_rmse(image_gt, image_fused):
    image_gt = image_gt.astype(np.float64)   ;   image_fused = image_fused.astype(np.float64)
    return np.sqrt(compute_mse(image_gt, image_fused))


def compute_nrmse(image_gt, image_fused):
    image_gt = image_gt.astype(np.float64)   ;   image_fused = image_fused.astype(np.float64)
    return np.sqrt(np.sum((image_gt - image_fused) ** 2) / np.sum(image_gt ** 2))
