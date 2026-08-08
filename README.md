# diffusion-model-for-mri-us-image-fusion

## Abstract
This repository presents two image fusion methods for Magnetic Resonance Imaging (MRI) and Ultrasound (US) images. The objective of combining these two imaging modalities is to exploit their complementary strengths, namely the high spatial resolution of US images and the high tissue contrast of MRI images, while mitigating their respective limitations: the high noise level and low contrast of US images, and the relatively low spatial resolution of MRI images.

The first method, located in the PALM-main directory, is an iterative optimization algorithm based on the PALM framework, specifically designed for MRI and US image fusion **[1]**. This implementation extends and improves a previous work, available on GitHub at: [Denoising Diffusion model with Proximal Alternating Linearized Minimization for image fusion](https://github.com/TLongin/Denoising-Diffusion-model-with-Proximal-Alternating-Linearized-Minimization) (available on the PALM directory), by providing a faithful Python translation of the original MATLAB code and introducing several new features.

The second method, located in the DDFM-PALM-main directory, is a novel MRI and US image fusion framework based on diffusion models. It was developed by drawing inspiration from the Denoising Diffusion Model for Multi-Modality Image Fusion (DDFM), introduced in **[2]**, whose official implementation is available on GitHub: [DDFM: Denoising Diffusion Model for Multi-Modality Image Fusion](https://github.com/Zhaozixiang1228/MMIF-DDFM).
The original DDFM method is a diffusion-based image fusion framework designed for infrared and visible image fusion. However, these imaging modalities differ significantly from MRI and US images. Therefore, we adapted the DDFM method to account for the specific characteristics of MRI and US modalities, notably by incorporating the PALM algorithm presented previously into the diffusion process. 
Our method is a continuation of previous work carried out within the same research group, available on GitHub at: [Denoising Diffusion model with Proximal Alternating Linearized Minimization for image fusion](https://github.com/TLongin/Denoising-Diffusion-model-with-Proximal-Alternating-Linearized-Minimization). This work aims to improve and correct the previous implementation.

## Results




## Usage
### 1. Installation Procedure
We recommend following the instructions provided in the Github repository Denoising Diffusion Model for Multi-Modality Image fusion, as the original code (which we modified for our needs) comes from there [2]. Follow the instructions provided with the files given in our repository. Please note that we used Python version 3.12 and not 3.8. It is advisable to install the packages listed in the Github repository (requirement.txt file) one by one and not all at once with the command indicated :
```python
pip install requirements.txt
```
without specifying the version so that any dependency issues between different packages are automatically resolved. This process is long and tedious, but it is the only way to ensure that all of the packages have the correct versions, without dependency issues.
Please do not forget to download the checkpoint "256x256_diffusion_uncond.pt" available at [this link](https://github.com/openai/guided-diffusion)  and place it in the './DDFM-PALM-main/models/' directory. 

### 2. Inference
If you want to infer with our method DDFM-PALM, please go to the selected dossier and run
```python
python sampleTLSE_unidim.py
```
for data which have se ame size, n'oublier pas de selectionner


## References
**[1]** Oumaima El Mansouri, Fabien Vidal, Adrian Basarab, Pierre Payoux, Denis Kouamé, and Jean-Yves Tourneret. Fusion of magnetic resonance and ultrasound images for endometriosis detection. IEEE Transactions on Image Processing, 2020.

**[2]** Zixiang Zhao, Haowen Bai, Yuanzhi Zhu, Jiangshe Zhang, Shuang Xu, Yulun Zhang, Kai Zhang, Deyu Meng, Radu Timofte, and Luc Van Gool. Ddfm : Denoising diffusion model for multi-modality image fusion, 2023. 


If you use this code, please cite:

```bibtex
@article{9018380,
  author={El Mansouri, Oumaima and Vidal, Fabien and Basarab, Adrian and Payoux, Pierre and Kouamé, Denis and Tourneret, Jean-Yves},
  journal={IEEE Transactions on Image Processing},
  title={Fusion of Magnetic Resonance and Ultrasound Images for Endometriosis Detection}, 
  year={2020},
  volume={29},
  number={},
  pages={5324-5335},
  keywords={Spatial resolution;Magnetic resonance imaging;Image fusion;Diseases;Magnetic resonance;Image fusion;magnetic resonance imaging;ultrasound imaging;super-resolution;despeckling;proximal alternating linearized minimization},
  doi={10.1109/TIP.2020.2975977}
}

@InProceedings{Zhao_2023_ICCV,
  author    = {Zhao, Zixiang and Bai, Haowen and Zhu, Yuanzhi and Zhang, Jiangshe and Xu, Shuang and Zhang, Yulun and Zhang, Kai and Meng, Deyu and Timofte, Radu and Van Gool, Luc},
  title     = {DDFM: Denoising Diffusion Model for Multi-Modality Image Fusion},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2023},
  pages     = {8082-8093}
}
```

















