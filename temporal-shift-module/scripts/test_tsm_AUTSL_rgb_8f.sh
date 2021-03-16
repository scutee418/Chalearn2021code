python test_models.py AUTSL \
    --weights=checkpoint/TSM_AUTSL_RGB_resnet50_shift8_blockres_avg_segment8_e50/ckpt.50pth.tar \
    --test_segments=8 --test_crops=10 \
    --batch_size=32 -j 4 \
    --test_list /home/GaoXiang/program_file/AUTSL/mmaction/data/AUTSL/AUTSL_file_list2.csv \
    --gpus 0 1 2