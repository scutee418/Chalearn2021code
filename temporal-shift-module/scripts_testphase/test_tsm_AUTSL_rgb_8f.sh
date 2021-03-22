python test_models.py AUTSL \
    --weights=checkpoint/TSM_AUTSL_RGB_resnet50_shift8_blockres_avg_segment8_e10/ckpt.10pth.tar \
    --test_segments=8 --test_crops=10 \
    --batch_size=24 -j 4 \
    --test_list ../mmaction/data/AUTSL/test/test_RGB_pse93.csv \
    --gpus 1 2 3
    --name TSM_addvalset_addtestsetv1_RGB_finetune_Epoch10