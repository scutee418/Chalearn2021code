# You should get TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
python main.py AUTSL RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.005 --wd 1e-4 --lr_steps 10 15 --epochs 20 \
     --batch-size 30 -j 4 --dropout 0.5 --consensus_type=avg --eval-freq=5 \
     --shift --shift_div=8 --shift_place=blockres --npb --gpus 0 1 2 \
     --tune_from checkpoint/old_checkpoints/TSM_AUTSL_RGB_resnet50_shift8_blockres_avg_segment8_e50/ckpt.50pth.tar