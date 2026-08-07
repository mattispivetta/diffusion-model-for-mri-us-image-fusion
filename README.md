# diffusion-model-for-mri-us-image-fusion

## Abstract
This repository presents two image fusion methods for Magnetic Resonance Imaging (MRI) and Ultrasound (US) images. The objective of combining these two imaging modalities is to exploit their complementary strengths, namely the high spatial resolution of US images and the high tissue contrast of MRI images, while mitigating their respective limitations: the high noise level and low contrast of US images, and the relatively low spatial resolution of MRI images.

The first method, located in the PALM-main directory, is an iterative optimization algorithm based on the PALM framework, specifically designed for MRI and US image fusion []. This implementation extends and improves a previous work, available on GitHub at: [Denoising Diffusion model with Proximal Alternating Linearized Minimization for image fusion](https://github.com/TLongin/Denoising-Diffusion-model-with-Proximal-Alternating-Linearized-Minimization) (available on the PALM directory), by providing a faithful Python translation of the original MATLAB code and introducing several new features.

The second method, located in the DDFM-PALM-main directory, is a novel MRI and US image fusion framework based on diffusion models. It was developed by drawing inspiration from the Denoising Diffusion Model for Multi-Modality Image Fusion (DDFM), introduced in [], whose official implementation is available on GitHub: [DDFM: Denoising Diffusion Model for Multi-Modality Image Fusion](https://github.com/Zhaozixiang1228/MMIF-DDFM).
The original DDFM method is a diffusion-based image fusion framework designed for infrared and visible image fusion. However, these imaging modalities differ significantly from MRI and US images. Therefore, we adapted the DDFM method to account for the specific characteristics of MRI and US modalities, notably by incorporating the PALM algorithm presented previously into the diffusion process. 
Our method is a continuation of previous work carried out within the same research group, available on GitHub at: [Denoising Diffusion model with Proximal Alternating Linearized Minimization for image fusion](https://github.com/TLongin/Denoising-Diffusion-model-with-Proximal-Alternating-Linearized-Minimization). This work aims to improve and correct the previous implementation.
























