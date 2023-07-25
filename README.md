# Boosting Single Image Super-Resolution via Partial Channel Shifting
This repository is for Partial Channel Shifting (PCS) introduced in the following paper "Boosting Single Image Super-Resolution via Partial Channel Shifting", and it is submitted to ICCV 2023.


The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and test on Ubuntu 18.04 environment (Python3.6, PyTorch = 1.1.0) with TitanXP GPUs. 
## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Visualization](#Visualization)
5. [Acknowledgements](#acknowledgements)

## Introduction

Although deep learning has significantly facilitated the progress of single image super-resolution (SISR) in recent years, it still hits bottlenecks to further improve SR performance with the continuous growth of model scale. Therefore, one of the hotspots in the field is to construct efficient SISR models by elevating the effectiveness of feature representation. In this work, we present a straightforward and generic approach for feature enhancement that can effectively promote the performance of SR models, dubbed partial channel shifting (PCS). Specifically, it is inspired by the temporal shifting in video understanding and displaces part of the channels along the spatial dimensions, thus allowing the effective receptive field to be amplified and the feature diversity to be augmented at almost zero cost. Also, it can be assembled into off-the-shelf models as a plug-and-play component for performance boosting without extra network parameters and computational overhead. However, regulating the features with PCS encounters some issues, like shifting directions and amplitudes, proportions, and patterns of shifted channels, etc. We impose some technical constraints on the issues to simplify the general channel shifting. Extensive and throughout experiments illustrate that the PCS indeed enlarges the effective receptive field, augments the feature diversity for efficiently enhancing SR recovery, and can endow obvious performance gains to existing models.

### Partial Channel Shifting

![PCS](/pictures/pcs.png)

### DRRN (PCS)

![DRRN (PCS)](/pictures/drrn_pcs.png)

### EDSR (PCS)

![EDSR (PCS)](/pictures/edsr_pcs.png)



## Train
### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

2. Specify '--dir_data' based on the HR and LR images path. 

For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Begin to train

1. (optional) Download pretrained models for our paper.

    Pre-trained models can be downloaded from [Google Drive](https://drive.google.com/file/d/1-gUHqcmPaW_l3JKeNfRbCCeNFTuZxq58/view?usp=share_link) 

2. Cd to 'src', run the following script to train models.

    **For Training**

    ```bash
    #Uni-directional Shift
    #EDSR (S, PCS), gamma=1/8, |h|=2, |w|=0
    python3 main.py --model edsr_shift_uni --scale 2 --move_c 8 --move_p 2 --patch_size 96 --save EDSR_Uni_PCS_C8_P2_X2
    
    #Bi-directional Shift
    #DRRN (PCS), gamma=1/16, |h|=2, |w|=0
    python3 main.py --template DRRN_S --move_c 4 --move_p 2 --scale 2 --patch_size 96 --save DRRN_Bi_PCS_C4_P2_X2
    python3 main.py --template DRRN_S --move_c 4 --move_p 2 --scale 3 --patch_size 96 --save DRRN_Bi_PCS_C4_P2_X3 --pre_train DRRN_PCS_C4_P2_X2/model/model_best.pt
    python3 main.py --template DRRN_S --move_c 4 --move_p 2 --scale 3 --patch_size 96 --save DRRN_Bi_PCS_C4_P2_X4 --pre_train DRRN_PCS_C4_P2_X3/model/model_best.pt
    #EDSR (S, PCS), gamma=1/8, |h|=2, |w|=0
    python3 main.py --model edsr_shift_bi --scale 2 --move_c 4 --move_p 2 --patch_size 96 --save EDSR_Bi_PCS_C4_P2_X2
    python3 main.py --model edsr_shift_bi --scale 3 --move_c 4 --move_p 2 --patch_size 144 --save EDSR_Bi_PCS_C4_P2_X3 --pre_train EDSR_PCS_C4_P2_X2/model/model_best.pt
    python3 main.py --model edsr_shift_bi --scale 4 --move_c 4 --move_p 2 --patch_size 192 --save EDSR_Bi_PCS_C4_P2_X4 --pre_train EDSR_PCS_C4_P2_X3/model/model_best.pt

    #EDSR (L, PCS) gamma=1/16, |h|=2, |w|=0
    python3 main.py --model edsr_shift_bi --scale 2 --move_c 8 --move_p 2  --n_resblocks 32 --n_feats 256 --res_scale 0.1 --patch_size 96 --save EDSR_Bi_PCS_C4_P2_X2_L
    python3 main.py --model edsr_shift_bi --scale 3 --move_c 8 --move_p 2  --n_resblocks 32 --n_feats 256 --res_scale 0.1 --patch_size 144 --save EDSR_Bi_PCS_C4_P2_X3_L --pre_train EDSR_PCS_C4_P2_X2_L/model/model_best.pt
    python3 main.py --model edsr_shift_bi --scale 4 --move_c 8 --move_p 2  --n_resblocks 32 --n_feats 256 --res_scale 0.1 --patch_size 192 --save EDSR_Bi_PCS_C4_P2_X4_L --pre_train EDSR_PCS_C4_P2_X3_L/model/model_best.pt
    
    #Cross-directional Shift
    #EDSR (S, PCS), gamma=1/8, |h|=2, |w|=0
    python3 main.py --model edsr_shift_cross --scale 2 --move_c 4 --move_p 2 --patch_size 96 --save EDSR_Cross_PCS_C4_P2_X2
    
    #Quad-directional Shift
    #EDSR (S, PCS), gamma=1/8, |h|=2, |w|=0
    python3 main.py --model edsr_shift_quad --scale 2 --move_c 2 --move_p 2 --patch_size 96 --save EDSR_Quad_PCS_C2_P2_X2

    #RFDN
    python main.py --model rfdn --scale 2 --data_test Set5+Set14+B100+Urban100+Manga109 --lr 5e-4 --save RFDNx2 
    python main.py --model rfdn --scale 4 --patch_size 192 --data_test Set5+Set14+B100+Urban100+Manga109 --lr 5e-4 --save RFDNx4 
   
    #RFDN(PCS)
    python main.py --model rfdn_pcs --scale 2 --data_test Set5+Set14+B100+Urban100+Manga109 --lr 5e-4 --save RFDN_PCSx2
    python main.py --model rfdn_pcs --scale 4 --patch_size 192 --data_test Set5+Set14+B100+Urban100+Manga109 --lr 5e-4 --save RFDN_PCSx4
   
    #BSRN
    python main.py --model bsrn --scale 2 --data_test Set5+Set14+B100+Urban100+Manga109 --lr 1e-3 --save BSRNx2 
    python main.py --model bsrn --scale 4 --patch_size 192 --data_test Set5+Set14+B100+Urban100+Manga109 --lr 1e-3 --save BSRNx4 
    
    #BSRN_PCS
    python main.py --model bsrn_pcs --scale 2 --data_test Set5+Set14+B100+Urban100+Manga109 --lr 1e-3 --save BSRN_PCSx2
    python main.py --model bsrn_pcs --scale 4 --patch_size 192 --data_test Set5+Set14+B100+Urban100+Manga109 --lr 1e-3 --save BSRN_PCSx4
   
    #VAPSR
    python main.py --model vapsr --scale 2 --data_test Set5+Set14+B100+Urban100+Manga109 --lr 5e-4 --save VAPSRx2 
    python main.py --model vapsr --scale 4 --patch_size 192 --data_test Set5+Set14+B100+Urban100+Manga109 --lr 5e-4 --save VAPSRx4 
   
    #VAPSR_PCS
    python main.py --model vapsr_pcs --scale 2 --data_test Set5+Set14+B100+Urban100+Manga109 --lr 5e-4 --save VAPSR_PCSx2 
    python main.py --model vapsr_pcs --scale 4 --patch_size 192 --data_test Set5+Set14+B100+Urban100+Manga109 --lr 5e-4 --save VAPSR_PCSx4 
   
    #OISR-LF
    python main.py --model oisr_lf --n_resblocks 8 --n_feats 122 --scale 2 --data_test Set5+Set14+B100+Urban100+Manga109 --lr 1e-4 --save OISR_LFx2 
    python main.py --model oisr_lf --n_resblocks 8 --n_feats 122 --scale 4 --patch_size 192 --data_test Set5+Set14+B100+Urban100+Manga109 --lr 1e-4 --save OISR_LFx4 
   
    #OISR-LF-PCS
    python main.py --model oisr_lf_pcs --n_resblocks 8 --n_feats 122 --scale 2 --data_test Set5+Set14+B100+Urban100+Manga109 --lr 1e-4 --save OISR_LF_PCSx2 
    python main.py --model oisr_lf_pcs --n_resblocks 8 --n_feats 122 --scale 4 --patch_size 192 --data_test Set5+Set14+B100+Urban100+Manga109 --lr 1e-4 --save OISR_LF_PCSx4 
   
    #OISR-RK3
    python main.py --model oisr --n_resblocks 22 --n_feats 256 --scale 2 --data_test Set5+Set14+B100+Urban100+Manga109 --lr 1e-4 --save OISR_x2 
    
    #OISR-RK3-PCS
    python main.py --model oisr_pcs --n_resblocks 8 --n_feats 122 --scale 2 --data_test Set5+Set14+B100+Urban100+Manga109 --lr 1e-4 --save OISR_PCSx2 
    
    #SAN
    python main.py --template SAN --scale 2 --data_test Set5+Set14+B100 --chop --lr 1e-4 --save SAN_x2 
   
    #SAN_PCS
    python main.py --template SAN_PCS --scale 2 --data_test Set5+Set14+B100 --chop --lr 1e-4 --save SAN_PCS_x2
   
    #NLSN
    python main.py --template NLSN --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 2 --data_test Set5+Set14+B100+Urban100+Manga109 --chop --lr 1e-4 --save NLSN_x2
   
    #NLSN_PCS
    python main.py --template NLSN_PCS --chunk_size 144 --n_hashes 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 2 --data_test Set5+Set14+B100+Urban100+Manga109 --chop --lr 1e-4 --save NLSN_PCS_x2 
    
   ```

## Test
1. Download benchmark datasets from [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/benchmark.tar)

2. (optional) Download pretrained models for our paper.

    All the models can be downloaded from [Google Drive](https://drive.google.com/file/d/1-gUHqcmPaW_l3JKeNfRbCCeNFTuZxq58/view?usp=share_link) 

3. Quantitative results on benchmark datasets.

| **Method**    | **Scale** | **Params.** | **Set5 PSNR** | **Set5 SSIM** | **Set14 PSNR** | **Set14 SSIM** | **B100 PSNR** | **B100 SSIM** | **Urban100 PSNR** | **Urban100 SSIM** | **Manga109 PSNR** | **Manga109 SSIM** |
|---------------|-----------|-------------|---------------|---------------|----------------|----------------|---------------|---------------|-------------------|-------------------|-------------------|-------------------|
| DRRN(PCS)     | 2         | 0.30M       | 37.81         | 0.9597        | 33.36          | 0.9155         | 32.04         | 0.8977        | 31.51             | 0.9216            | 38.08             | 0.9759            |
| EDSR(S, PCS)  | 2         | 1.37M       | 38.01         | 0.9605        | 33.60          | 0.9175         | 32.19         | 0.8997        | 32.17             | 0.9287            | 38.61             | 0.9770            |
| EDSR(S, PCS)+ | 2         | 1.37M       | 38.11         | 0.9608        | 33.70          | 0.9183         | 32.23         | 0.9002        | 32.34             | 0.9302            | 38.84             | 0.9776            |
| EDSR(L, PCS)  | 2         | 40.73M      | 38.24         | 0.9613        | 34.04          | 0.9209         | 32.36         | 0.9019        | 32.99             | 0.9361            | 39.24             | 0.9783            |
| EDSR(L, PCS)+ | 2         | 40.73M      | 38.30         | 0.9616        | 34.11          | 0.9212         | 32.41         | 0.9024        | 33.17             | 0.9375            | 39.44             | 0.9786            |
| DRRN(PCS)     | 3         | 0.30M       | 34.16         | 0.9252        | 30.10          | 0.8374         | 28.95         | 0.8008        | 27.70             | 0.8411            | 32.95             | 0.9405            |
| EDSR(S, PCS)  | 3         | 1.55M       | 34.41         | 0.9272        | 30.33          | 0.8424         | 29.11         | 0.8055        | 28.23             | 0.8541            | 33.59             | 0.9448            |
| EDSR(S, PCS)+ | 3         | 1.55M       | 34.53         | 0.9282        | 30.44          | 0.8440         | 29.17         | 0.8067        | 28.39             | 0.8567            | 33.92             | 0.9466            |
| EDSR(L, PCS)  | 3         | 43.68M      | 34.76         | 0.9299        | 30.56          | 0.8469         | 29.28         | 0.8101        | 28.88             | 0.8672            | 34.21             | 0.9489            |
| EDSR(L, PCS)+ | 3         | 43.68M      | 34.85         | 0.9305        | 30.71          | 0.8488         | 29.35         | 0.8112        | 29.09             | 0.8703            | 34.57             | 0.9506            |
| DRRN(PCS)     | 4         | 0.30M       | 31.96         | 0.8921        | 28.38          | 0.7769         | 27.42         | 0.7307        | 25.65             | 0.7707            | 29.86             | 0.9007            |
| EDSR(S, PCS)  | 4         | 1.52M       | 32.20         | 0.8949        | 28.61          | 0.7826         | 27.58         | 0.7361        | 26.13             | 0.7881            | 30.51             | 0.9087            |
| EDSR(S, PCS)+ | 4         | 1.52M       | 32.36         | 0.8970        | 28.72          | 0.7849         | 27.65         | 0.7379        | 26.29             | 0.7918            | 30.85             | 0.9122            |
| EDSR(L, PCS)  | 4         | 43.10M      | 32.51         | 0.8989        | 28.83          | 0.7881         | 27.72         | 0.7423        | 26.69             | 0.8053            | 31.10             | 0.9167            |
| EDSR(L, PCS)+ | 4         | 43.10M      | 32.67         | 0.9006        | 28.96          | 0.7904         | 27.82         | 0.7442        | 26.92             | 0.8099            | 31.52             | 0.9199            |

4. Cd to 'src', run the following scripts. To train BSRN and VAPSR, a higher PyTorch version is required (>1.8) as they use nn.GELU().

    **For Inference**

    ```bash
    #Bi-directional Shift
    #DRRN (PCS), gamma=1/16, |h|=2, |w|=0
    python3 main.py --template DRRN_S --scale 2 --move_c 4 --move_p 2 --data_test Set5+Set14+B100+Urban100+Manga109 --pre_train ../model_zoo/drrn_pcs_c4_p2_x2.pt --test_only
    python3 main.py --template DRRN_S --scale 3 --move_c 4 --move_p 2 --data_test Set5+Set14+B100+Urban100+Manga109 --pre_train ../model_zoo/drrn_pcs_c4_p2_x3.pt --test_only
    python3 main.py --template DRRN_S --scale 4 --move_c 4 --move_p 2 --data_test Set5+Set14+B100+Urban100+Manga109 --pre_train ../model_zoo/drrn_pcs_c4_p2_x4.pt --test_only
    
    #EDSR (S, PCS), , gamma=1/8, |h|=2, |w|=0
    python3 main.py --model edsr_shift_bi --scale 2 --move_c 4 --move_p 2 --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --pre_train ../model_zoo/edsr_pcs_c4_p2_x2.pt
    python3 main.py --model edsr_shift_bi --scale 3 --move_c 4 --move_p 2 --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --pre_train ../model_zoo/edsr_pcs_c4_p2_x3.pt
    python3 main.py --model edsr_shift_bi --scale 4 --move_c 4 --move_p 2 --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --pre_train ../model_zoo/edsr_pcs_c4_p2_x4.pt

    #EDSR (L, PCS), , gamma=1/16, |h|=2, |w|=0
    python3 main.py --model edsr_shift_bi --scale 2 --move_c 8 --move_p 2  --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --pre_train ../model_zoo/edsr_pcs_c8_p2_x2_large.pt
    python3 main.py --model edsr_shift_bi --scale 3 --move_c 8 --move_p 2  --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --pre_train ../model_zoo/edsr_pcs_c8_p2_x3_large.pt
    python3 main.py --model edsr_shift_bi --scale 4 --move_c 8 --move_p 2  --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --pre_train ../model_zoo/edsr_pcs_c8_p2_x4_large.pt
      
    #Test with Self-ensemble
    python3 main.py --model edsr_shift_bi --scale 2 --move_c 8 --move_p 2  --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --pre_train ../model_zoo/edsr_pcs_c8_p2_x2_large.pt --self
    ```

## Visualization
The visualization of *Effective Receptive Fields (ERFs)* is generated based on the repository of [RepLKNet](https://github.com/DingXiaoH/RepLKNet-pytorch) by Ding *et al*.

The visualization of *Local Attribute Map (LAM)* is generated based on the repository of [LAM](https://github.com/X-Lowlevel-Vision/LAM_Demo) by Gu *et al*.

## Acknowledgements
This code is built on the framework of [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for their open-source codes.
