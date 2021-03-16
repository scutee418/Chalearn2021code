python main.py AUTSL Flow \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.005 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
     --batch-size 32 -j 4 --dropout 0.5 --consensus_type=avg --eval-freq=5 \
     --shift --shift_div=8 --shift_place=blockres --npb --gpus 0 1 2
     --tune_from TSM_kinetics_Flow_resnet50_shift8_blockres_avg_segment8_e50.pth