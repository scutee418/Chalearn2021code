model = dict(
    type='TSN3D',
    backbone=dict(
        type='ResNet_I3D',
        pretrained='modelzoo://resnet50',
        depth=50,
        num_stages=4,
        out_indices=[3],
        frozen_stages=-1,
        inflate_freq=(0, 0, 1, 1),
        conv1_kernel_t=1,
        conv1_stride_t=1,
        pool1_kernel_t=1,
        pool1_stride_t=1,
        inflate_style='3x1x1',
        bn_eval=False,
        no_pool2=True,
        partial_bn=False,
        style='pytorch'),
    spatial_temporal_module=dict(
        type='SimpleSpatialTemporalModule',
        spatial_type='avg',
        temporal_size=8,
        spatial_size=7),
    segmental_consensus=dict(
        type='SimpleConsensus',
        consensus_type='avg'),
    cls_head=dict(
        type='ClsHead',
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.5,
        in_channels=2048,
        num_classes=226))
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'RawFramesDataset'
data_root = '/home/GaoXiang/program_file/AUTSL/mmaction/data/AUTSL'
data_val_root = '/home/GaoXiang/program_file/AUTSL/mmaction/data/AUTSL/test/rawframes_align'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    videos_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file='/home/GaoXiang/program_file/AUTSL/mmaction/data/AUTSL/train/train_val_depth_rawframes.csv',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        input_format="NCTHW",
        num_segments=1,
        new_length=8,
        new_step=8,
        random_shift=True,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=256,
        input_size=224,
        div_255=False,
        flip_ratio=0.5,
        resize_crop=True,
        resize_keep_ratio=True,
        oversample=None,
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=True,
        test_mode=False,
        mask=False,
        bbox_label="bboxexpand_label_train_val_depth.csv",
        afterdetect_resize=(280,320),
        RandomResizeCrop_scale=(0.2, 1.0)
        ),
    val=dict(
        type=dataset_type,
        ann_file='/home/GaoXiang/program_file/AUTSL/mmaction/data/AUTSL/test/test_depth_pse93.csv',
        img_prefix=data_val_root,
        img_norm_cfg=img_norm_cfg,
        input_format="NCTHW",
        num_segments=1,
        new_length=8,
        new_step=8,
        random_shift=False,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=256,
        input_size=224,
        div_255=False,
        flip_ratio=0,
        resize_keep_ratio=True,
        oversample=None,
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=False,
        test_mode=True,
        mask=False,
        bbox_label="bboxexpand_label_test_depth.csv",
        afterdetect_resize=(280,320),
        RandomResizeCrop_scale=(0.8, 1.0)
        ),
   test=dict(
        type=dataset_type,
        ann_file='/home/GaoXiang/program_file/AUTSL/mmaction/data/AUTSL/test/test_depth_pse93.csv',
        img_prefix=data_val_root,
        img_norm_cfg=img_norm_cfg,
        input_format="NCTHW",
        num_segments=10,
        new_length=8,
        new_step=8,
        random_shift=True,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=256,
        input_size=256,
        div_255=False,
        flip_ratio=0,
        resize_keep_ratio=True,
        oversample='three_crop',
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=False,
        test_mode=True,
        mask=False,
        bbox_label="bboxexpand_label_test_depth.csv"
        ))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[30,45],
    warmup_ratio=0.01,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=10)

checkpoint_config = dict(interval=1)
workflow = [('train', 10),('val', 1)]
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 55
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs_testphase/slowonly_addvalset_detect_lr0.01_cropratio0.2_epoch55_depth'
load_from = None
resume_from = None

'''
加检测，bbox扩充，加验证集一起训练
'''