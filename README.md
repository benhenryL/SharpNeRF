# Sharp-NeRF: Grid-based Fast Deblurring Neural Radiance Fields Using Sharpness Prior

Byeonghyeon Lee*, Howoong Lee*, Usman Ali, and Eunbyung Park

[Project Page](https://benhenryl.github.io/SharpNeRF/) &nbsp; [Paper](https://arxiv.org/abs/2401.00825/)

Our code is based on TensoRF (https://github.com/apchenstu/TensoRF).


## 1. Requirements

```
conda create -n SharpNeRF python=3.8
conda activate SharpNeRF

// Please install pytorch compatible to your cuda version.
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install six tqdm scikit-image==0.19.2 opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard setuptools==59.5.0 plyfile
```

We tested our model with the following enivornment:
- Ubuntu 18.04 
- Python 3.8.0
- Pytorch 1.10.0 
- Cuda 11.4

## 2. Dataset
Download 'real_defocus_blur' and 'synthetic_defocus_blur' dataset at 
https://hkustconnect-my.sharepoint.com/:f:/g/personal/lmaag_connect_ust_hk/EqB3QrnNG5FMpGzENQq_hBMBSaCQiZXP7yGCVlBHIGuSVA?e=UaSQCC  
Then, the dataset directory should look like below:
```
data/
    real_defocus_blur/
        defocusbush/
        ...
        defocustools/
            images/
            images_4/
            sparse/
            ...
    synthetic_defocus_blur/
        defocuscozy2room/
            images/
            images_4/
            sparse/
            ...
        ...
```


## 3. Preprocess
Preprocess stage for computing sharpness level consists of two substeps: depth map estimation and focus map estimation.  
You can download the precomputed one at https://drive.google.com/file/d/1IzucVQsnzA-fYw2LdB8odur1DHUIaNlO/view?usp=sharing  
Unzip the downloaded file and replace the 'preprocess' directory with the downloaded files.  

Or, you can compute it with the following steps.  


### 3-1. Depth map estimation
We use MiDas (https://github.com/isl-org/MiDaS) with dpt_beit_large_512 weight to estimate scene depth.  
Please run MiDas to compute depth map.  
Then run 
```
python preprocess_depth.py PATH_TO_OUTPUT SCENE_NAME
// ex. python preprocess_depth.py ./MiDaS/output tools
```
cf. You can process depth map of all scenes by setting SCENE_NAME as 'all'
```
python preprocess_depth.py ./MiDaS/output all // Process depth map of all scenes, including both real and synthetic dataset.
```

### 3-2. Focus map estimation
We use DMENet (https://github.com/codeslake/DMENet) to estimate focus map.  
Download pretrained weight (dmenet_pretrained.npz) at https://drive.google.com/file/d/1wAahb4D8ldjigvPxYuFNFJOZh3Y29B_C/view?usp=drive_link  and move it to the 'preprocess' directory.  
Then run 
```
python preprocess_focus.py PATH_TO_DATASET SCENE_NAME
// ex. python preprocess_focus.py ./data tools
// cf. python preprocess_focus.py ./data all
```

### 3-3. Compute sharpness level
Now you can compute sharpness level.
```
python preprocess_sharpness_level.py SCENE_NAME
// ex. python preprocess_sharpness_level.py tools
// cf. python preprocess_sharpness_level.py all
```

## 4. Training
Now you can run SharpNeRF with following command:
```
bash run.sh GPU_ID DATA_PATH EXP_NAME
// ex. bash run.sh 0 ./data/real_defocus_blur/defocustools test
```
