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


Example:
train:
We use the config file "config/test_phase/slowonly_addvalset_addtestsetv1_lr0.01_cropratio0.08_epoch83_depth.py" if we train a model using whole image as input
We use the config file "config/test_phase/slowonly_addvalset_addtestsetv1_detect_lr0.01_cropratio0.2_epoch95_RGB.py" if we train a model using person-cropped image as input
test:
We use the script "tools/dist_test_recognizer.sh" and input the config file "config/test_phase/slowonly_addvalset_addtestsetv1_lr0.01_cropratio0.08_epoch83_depth.py" and model parameters "work_dirs_testphase/slowonly_addvalset_addtestsetv1_lr0.01_cropratio0.08_epoch83_depth/epoch_71.pth" and finally get the prediction result in folder val_result