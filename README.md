#### **For train:**

- train a slowonly model using whole image as input

```
bash mmaction/tools/dist_train_recognizer.sh mmaction/config/test_phase/slowonly_addvalset_addtestsetv1_lr0.01_cropratio0.08_epoch83_depth.py 3
```
- train a slowonly model using person-cropped image as input
- 
```
bash mmaction/tools/dist_train_recognizer.sh python mmaction/config/test_phase/slowonly_addvalset_addtestsetv1_detect_lr0.01_cropratio0.2_epoch95_RGB.py 3
```
- train a slowfast_nln_multigrid model using depth images as input

```
# 1. modify line 81 of slowfast/datasets/kinects.py  "{}.csv" to "{}_depth.csv"
# 2. cd to root path of slowfast
python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_NLN_8x8_R50.yaml DATA.PATH_TO_DATA_DIR PathOfData MULTIGRID.SHORT_CYCLE True MULTIGRID.LONG_CYCLE True
```
- train a slowfast_multigrid model using optical flowx images as input

```
# 1. modify line 81 of slowfast/datasets/kinects.py  "{}.csv" to "{}_flowx.csv"
# 2. cd to root path of slowfast
python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_8x8_R50_stepwise_multigrid.yaml DATA.PATH_TO_DATA_DIR PathOfData
```
- train a slowfast_nln_multigrid_256 model 

```
# 1. cd to root path of slowfast
python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_8x8_R50_stepwise_multigrid.yaml DATA.PATH_TO_DATA_DIR PathOfData DATA.TRAIN_CROP_SIZE 256 TRAIN_JITTER_SCALES [260, 320]
```


#### **For test:**

- slowonly
```
We use the script "tools/dist_test_recognizer.sh" and input the config file "config/test_phase/slowonly_addvalset_addtestsetv1_lr0.01_cropratio0.08_epoch83_depth.py" and model parameters "work_dirs_testphase/slowonly_addvalset_addtestsetv1_lr0.01_cropratio0.08_epoch83_depth/epoch_71.pth" and finally get the prediction result in folder val_result
```
- slowfast

```
# 1. cd to root path of slowfast
python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_8x8_R50_stepwise_multigrid.yaml DATA.PATH_TO_DATA_DIR PathOfData TRAIN.CHECKPOINT_FILE_PATH checkpoints/checkpoint_epoch_00254.pyth TRAIN.ENABLE False
```
#### ** Dataset:**


```
In test phase we use the config file in:
configs/test_phase/*
where:
data root is the path to train data, which is organized as:
train
|——optical_flow_depth
|——optical_flow_RGB
|       |——0
|       |  |——signer0_sample431_color
|       |  |           |——flow_x_00001.jpg
|       |  |           |——flow_x_00002.jpg
|       |  |           |——flow_x_00003.jpg
|       |  |           |——    .
|       |  |           |——    .
|       |  |           |——    .
|       |  |           |——flow_y_00001.jpg
|       |  |           |——flow_y_00002.jpg
|       |  |           |——flow_y_00003.jpg
|       |  |           |——    .
|       |  |           |——    .
|       |  |           |——    .
|       |  |——signer0_sample515_color
|       |  |——         .
|       |  |——         .
|       |  |——         .
|       |——1
|       |——2
|       . 
|       . 
|       . 
|       |——226
|——rawframes_align_depth
|——rawframes_align_RGB
|       |——0
|       |  |——signer0_sample431_color
|       |  |           |——img_00001.jpg
|       |  |           |——img_00002.jpg
|       |  |           |——img_00003.jpg
|       |  |           |——    .
|       |  |           |——    .
|       |  |           |——    .
|       |  |——signer0_sample515_color
|       |  |——         .
|       |  |——         .
|       |  |——         .
|       |——1
|       |——2
|       . 
|       . 
|       . 
|       |——226


and data_val_root is the path to val data or test data, which is organized as:

val
|——optical_flow
|       |——signer1_sample1_color
|       |        |——flow_x_00001.jpg
|       |        |——flow_x_00002.jpg
|       |        |——    .
|       |        |——    .
|       |——signer1_sample1_depth
|       |——signer1_sample2_depth
|       |——        .
|       |——        .
|       |——        .


all data is organized as :

AUTSL
|—— train
|—— test
|—— val


for slowfast, there is no need to change the path of the Chalearn dataset
```


