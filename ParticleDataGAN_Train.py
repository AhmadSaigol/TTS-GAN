#!/usr/bin/env bash

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, default="0")
    parser.add_argument('--node', type=str, default="0015")
    opt = parser.parse_args()

    return opt
#args = parse_args()
'''
os.system(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_GAN.py \
-gen_bs 16 \
-dis_bs 16 \
--dist-url 'tcp://localhost:4321' \
--dist-backend 'nccl' \
--world-size 1 \
--rank {args.rank} \
--dataset UniMiB \
--bottom_width 8 \
--max_iter 500000 \
--img_size 32 \
--gen_model my_gen \
--dis_model my_dis \
--df_dim 384 \
--d_heads 4 \
--d_depth 3 \
--g_depth 3 \
--dropout 0 \
--latent_dim 100 \
--gf_dim 1024 \
--num_workers 16 \
--g_lr 0.0001 \
--d_lr 0.0003 \
--optimizer adam \
--loss lsgan \
--wd 1e-3 \
--beta1 0.9 \
--beta2 0.999 \
--phi 1 \
--batch_size 16 \
--num_eval_imgs 50000 \
--init_type xavier_uniform \
--n_critic 1 \
--val_freq 20 \
--print_freq 50 \
--grow_steps 0 0 \
--fade_in 0 \
--patch_size 2 \
--ema_kimg 500 \
--ema_warmup 0.1 \
--ema 0.9999 \
--diff_aug translation,cutout,color \
--class_name Running \
--exp_name Running")
'''
os.system(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_GAN.py \
-gen_bs 512 \
-dis_bs 512 \
--world-size 1 \
--rank 0 \
--dataset particle_data \
--data_path "/home/ahmad/TimeGAN_tf2.16/timeGAN/data/PM2.5_bins_split_type_kfold_files_REFs_QTIs" \
--which_folder_data QTI \
--fold_no 1 \
--upsample_Y \
--bottom_width 8 \
--max_iter 5 \
--gen_model my_gen \
--dis_model my_dis \
--df_dim 384 \
--d_heads 4 \
--g_heads 4 \
--d_depth 3 \
--g_depth 3 \
--dropout 0 \
--latent_dim 100 \
--gf_dim 1024 \
--g_lr 0.0001 \
--d_lr 0.0003 \
--optimizer adam \
--loss lsgan \
--beta1 0.9 \
--beta2 0.999 \
--batch_size 512 \
--num_eval_imgs 50000 \
--init_type xavier_uniform \
--n_critic 1 \
--val_freq 20 \
--print_freq 50 \
--grow_steps 0 0 \
--patch_size 2 \
--ema_kimg 500 \
--ema_warmup 0.1 \
--ema 0.9999 \
--class_name particle_data \
--channels 8 \
--seq_len 300 \
--exp_name particle_data_run_default_setings")
