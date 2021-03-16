python test_models.py AUTSL \
    --weights=checkpoint/TSM_AUTSL_RGB_Flow_resnet50_shift8_blockres_avg_segment8_e50/ckpt.20pth.tar \
    --test_segments=25 --test_crops=1 \
    --batch_size=32 -j 4 \
    --test_list /data/GaoXiang/program_file/AUTSL/mmaction/data/AUTSL/val/test_RGB_opticalflow.csv \
    --gpus 0 1 2