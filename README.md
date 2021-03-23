- Now We have uploaded all the model and data to google drive and you can download to reproduce our result. 
- Also, we provide all the inference scores corresponding to each model in folder "inference_testphase" so that you can run inf_rgb.py and inf_rgbd.py to perform model fusion and get our result directly.
- If you want to know more detail about train, please refer to Train_schedule.md

### For Slowonly
#### **Step 1: Installation for mmaction**
- cd to root path of our mmaction and perform the installation following the instruction on the [mmaction](https://github.com/open-mmlab/mmaction/blob/master/INSTALL.md) 
- It is crucial to config the third_party (i.e. decord and dense_flow,etc.)
- Note that we make some change in loading the data (i.e. detect the bounding box)
#### **Step 2: Get the processed data for slowonly**
- Note that the data is almost 20G! We use the dense_flow[mmaction](https://github.com/open-mmlab/mmaction/blob/master/DATASET.md) to extract frames and optical flow from both RGB data and depth data 
- Google Drive of preprocessed optical flow : [link](https://drive.google.com/drive/folders/1Q19By0siCnap5T-YIKlv30nxhNDNgv0M?usp=sharing)
- Google Drive of preprocessed divided frames of test videos: [link](https://drive.google.com/file/d/1Eq9sZxn61YRK76jfqkmMrXcsFPuAaI5j/view?usp=sharing)
- Then put the two folder into path "data/AUTSL/test" so the data is organized as :
```
mmaction
   data
   |——AUTSL
   |    |——test
   |         |——test_depth_pse93.csv
   |         |——test_RGB_pse93.csv
   |         |——optical_flow
   |                 |——signer6_sample1_color
   |                            |——flow_x_00001.jpg
   |                            |——flow_x_00002.jpg
   |                            |——     ..
   |                            |——     ..
   |                 |——signer6_sample1_depth
   |                 |——       ..
   |                 |——       ..
   |         |——rawframes_align
   |                 |——signer6_sample1_color
   |                            |——img_00001.jpg
   |                            |——img_00002.jpg
   |                            |——     ..
   |                            |——     ..
   |                 |——signer6_sample1_depth
   |                 |——       ..
   |                 |——       ..
```
#### **Step 3: Download the model**
- Download our model for slowonly [slowonly_model](https://drive.google.com/drive/folders/11FI_ymLr-4iv8-9w_kDzxBivS2G8_UMk?usp=sharing) and make the folder into the path to mmaction
```
mmaction
   |——work_dirs_testphase
   |         |——slowonly_addvalset_addtestsetv1_detect_lr0.01_cropratio0.2_epoch65_depth
   |                                        |——epoch_60.pth
   |         |——          ..
   |         |——          ..
   |         |——          ..
   |         |——          ..
```
#### **Step 4: Perform the inference with slowonly**
- Inference Details
```
# cd to root path of slowfast

# 1. get slowonly_addvalset_addtestsetv1_detect_depth_epoch60.npy
bash inference_scripts/slowonly_detect_depth.sh
then we will get the inference score "slowonly_addvalset_addtestsetv1_detect_depth_epoch60.npy" and prediction result in folder "val_result"
Note that the inference score is used to perform the final model fusion 
We take 1 hour with 3 GTX1080TI to perform this inference

# 2. get slowonly_addvalset_addtestsetv1_detect_RGB_epoch92.npy
bash inference_scripts/slowonly_detect_RGB.sh

# 3. get slowonly_addvalset_addtestsetv1_depth_epoch81.npy
bash inference_scripts/slowonly_depth.sh

# 4. get slowonly_addvelset_addtestsetv1_lr_0.01_cropratio0.08_epoch87.npy
bash inference_scripts/slowonly_RGB.sh

# 5. get slowonly_addvalset_addtestsetv1_input256_inference288_epoch98.npy
bash inference_scripts/slowonly_RGB_input256_inference288.sh

```

- We provide these five inference score with slowonly model in folder "inference_testphase" for final model fusion 



### For TSM
#### **Step 1: Download the model for TSM**
- cd to root path of our temproal-shift-module and download the model for TSM : [TSM_model](https://drive.google.com/drive/folders/1OhOHSUaJnRQvPQiW0uwcHc12p52OCZLM?usp=sharing)
```
temproal-shift-module
   |——checkpoints
   |         |——TSM_AUTSL_Flow_resnet50_shift8_blockres_avg_segment8_e10_testv1
   |                                        |——ckpt.10pth.tar
   |         |——          ..
   |         |——          ..
```
#### **Step 2: Perform the inference with TSM**
- Inference Details
```
# cd to root path of slowfast

# 1. get TSM_addvalset_addtestsetv1_RGB_finetune_Epoch10.npy
First,modify line 107 of ops/dataset_config.py to the absolute path to mmaction/data/AUTSL/test/rawframes_align
Then,modify line 5 of scripts_testphase/test_tsm_AUTSL_rgb_8f.sh to the absolute path to mmaction/data/AUTSL/test/test_RGB_pse93.csv
Then,run bash scripts_testphase/test_tsm_AUTSL_rgb_8f.sh
Finally,we get the inference score and prediction result in the folder "val_result"

# 2. get TSM_addvalset_addtestsetv1_RGBflow_finetune_Epoch10.npy
First,modify line 117 of ops/dataset_config.py to the absolute path to mmaction/data/AUTSL/test/optical_flow
Then,modify line 5 of scripts_testphase/test_tsm_AUTSL_rgb_8f.sh to the absolute path to mmaction/data/AUTSL/test/test_RGB_pse93.csv
Then,run bash scripts_testphase/test_tsm_AUTSL_rgbflow_8f.sh
Finally,we get the inference score and prediction result in the folder "val_result"

# 3. get TSM_addvalset_addtestsetv1_depth_flow_Epoch10
First,modify line 117 of ops/dataset_config.py to the absolute path to mmaction/data/AUTSL/test/optical_flow
Then,modify line 5 of scripts_testphase/test_tsm_AUTSL_rgb_8f.sh to the absolute path to mmaction/data/AUTSL/test/test_depth_pse93.csv
Then,run scripts_testphase/test_tsm_AUTSL_depthflow_8f.sh
Finally,we get the inference score and prediction result in the folder "val_result"

```

- We provide these three inference score with TSM model in folder "inference_testphase" for final model fusion


### For SlowFast
#### **Step 1: Install SlowFast**
- Follow the install instruction on the [github](https://github.com/facebookresearch/SlowFast.git) of the slowfast
.
#### **Step 2: Get Human Segmentation Data of the Test Set**
- Method: [Unet-MobileNetV2](https://github.com/thuyngch/Human-Segmentation-PyTorch)
- Google Drive of preprocessed test set: [link](https://drive.google.com/drive/folders/1I9Cam2NwRbt_zkGnO_phquLrpT3iCy8R?usp=sharing)
#### **Step 3: Get Optical Flow Data of the Test Set**
- Method: 
- Google Drive of preprocessed test set: [link](https://drive.google.com/drive/folders/1I9Cam2NwRbt_zkGnO_phquLrpT3iCy8R?usp=sharing)
#### **Step 4: Inference Slowfast**
- Pretrained Models: Google Drive ([link](https://drive.google.com/drive/folders/1JPZ_v-kwVULGsgJYGRtHDVDQLKX4vVXf?usp=sharing))
- Inference Details
```
# cd to root path of slowfast

# 1. get results_9543.pkl
python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_NLN_8x8_R50.yaml DATA.PATH_TO_DATA_DIR PathOfData TRAIN.CHECKPOINT_FILE_PATH checkpoints/slowfast_nln_multigrid.pyth TRAIN.ENABLE False MULTIGRID.SHORT_CYCLE True MULTIGRID.LONG_CYCLE True TEST.SAVE_RESULTS_PATH  results_9543.pkl

# 2. get results.pkl
python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_8x8_R50_stepwise_multigrid.yaml DATA.PATH_TO_DATA_DIR PathOfData TRAIN.CHECKPOINT_FILE_PATH checkpoints/slowfast_multigrid_merge1.pyth TRAIN.ENABLE False TEST.SAVE_RESULTS_PATH  results.pkl

# 3. get results_y.pkl
# modify line 81 of slowfast/datasets/kinects.py  "{}.csv" to "{}_flow_y.csv"
python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_8x8_R50_stepwise_multigrid.yaml DATA.PATH_TO_DATA_DIR PathOfData TRAIN.CHECKPOINT_FILE_PATH checkpoints/slowfast_multigrid_flow_dev.pyth TRAIN.ENABLE False TEST.SAVE_RESULTS_PATH  results_y.pkl

# 4. get results256_pse3.pkl
python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_NLN_8x8_R50.yaml DATA.PATH_TO_DATA_DIR PathOfData TRAIN.CHECKPOINT_FILE_PATH checkpoints/slowfast_nln_multigrid_256_pse3.pyth TRAIN.ENABLE False MULTIGRID.SHORT_CYCLE True MULTIGRID.LONG_CYCLE True DATA.TRAIN_CROP_SIZE 256 TRAIN_JITTER_SCALES [260, 320] TEST.NUM_ENSEMBLE_VIEWS 30 TEST.SAVE_RESULTS_PATH  results256_pse3.pkl

# 5. get results_seg.pkl
# modify line 81 of slowfast/datasets/kinects.py  "{}.csv" to "{}_seg.csv"
python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_NLN_8x8_R50.yaml DATA.PATH_TO_DATA_DIR PathOfData TRAIN.CHECKPOINT_FILE_PATH checkpoints/slowfast_nln_multigrid_seg.pyth TRAIN.ENABLE False MULTIGRID.SHORT_CYCLE True MULTIGRID.LONG_CYCLE True TEST.SAVE_RESULTS_PATH  results_seg.pkl

# 6. get results256_flowx.pkl
# modify line 81 of slowfast/datasets/kinects.py  "{}.csv" to "{}_flow_x.csv"
python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_NLN_8x8_R50.yaml DATA.PATH_TO_DATA_DIR PathOfData TRAIN.CHECKPOINT_FILE_PATH checkpoints/slowfast_nln_multigrid_256_flowx.pyth TRAIN.ENABLE False MULTIGRID.SHORT_CYCLE True MULTIGRID.LONG_CYCLE True DATA.TRAIN_CROP_SIZE 256 TRAIN_JITTER_SCALES [260, 320] TEST.NUM_ENSEMBLE_VIEWS 30 TEST.SAVE_RESULTS_PATH  results256_flowx.pkl

# 7. get results_flowy.pkl
# modify line 81 of slowfast/datasets/kinects.py  "{}.csv" to "{}_flow_y.csv"
python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_NLN_8x8_R50.yaml DATA.PATH_TO_DATA_DIR PathOfData TRAIN.CHECKPOINT_FILE_PATH checkpoints/slowfast_nln_multigrid_flowy.pyth TRAIN.ENABLE False MULTIGRID.SHORT_CYCLE True MULTIGRID.LONG_CYCLE True TEST.SAVE_RESULTS_PATH  results_flowy.pkl

```

### Get Final results (model fusion)**

#### For RGB track:
```
python inf_rgb.py
```
- Then, we will get the predictions.csv in current path

#### For RGBD track:
```
python inf_rgbd.py
```
- Then, we will get the predictions.csv in current path