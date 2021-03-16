python main.py AUTSL Flow \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.005 --wd 1e-4 --lr_steps 6 8 --epochs 10 \
     --batch-size 32 -j 4 --dropout 0.5 --consensus_type=avg --eval-freq=2 \
     --shift --shift_div=8 --shift_place=blockres --npb --gpus 1 2 3 \
     --tune_from checkpoint/TSM_AUTSL_Flow_resnet50_shift8_blockres_avg_segment8_e35/ckpt.35pth.tar