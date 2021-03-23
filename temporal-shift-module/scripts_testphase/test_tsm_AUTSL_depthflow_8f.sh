python test_models.py AUTSL \
    --weights=checkpoint/TSM_AUTSL_Flow_resnet50_shift8_blockres_avg_segment8_e10/ckpt.10pth.tar \
    --test_segments=25 --test_crops=1 \
    --batch_size=24 -j 4 \
    --test_list ../mmaction/data/AUTSL/test/test_depth_pse93.csv \
    --gpus 1 2 3
    --name TSM_addvalset_addtestsetv1_depth_flow_Epoch10