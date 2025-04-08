# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--loca_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    #parser.add_argument('--seed', default=12345, type=int,
    #                    help='seed for initializing training. ')
    
    # NOTE: ADDED BY AHMAD
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument(
        '--max_epoch',
        type=int,
        default=200,
        help='number of epochs of training')
    parser.add_argument(
        '--max_iter',
        type=int,
        default=None,
        help='set the max iteration number')
    parser.add_argument(
        '-gen_bs',
        '--gen_batch_size',
        type=int,
        default=64,
        help='size of the batches')
    parser.add_argument(
        '-dis_bs',
        '--dis_batch_size',
        type=int,
        default=64,
        help='size of the batches')
    parser.add_argument(
        '-bs',
        '--batch_size',
        type=int,
        default=64,
        help='size of the batches to load dataset')
    parser.add_argument(
        '--g_lr',
        type=float,
        default=0.0002,
        help='adam: gen learning rate')
    parser.add_argument(
        '--wd',
        type=float,
        default=0,
        help='adamw: gen weight decay')
    parser.add_argument(
        '--d_lr',
        type=float,
        default=0.0002,
        help='adam: disc learning rate')
    parser.add_argument(
        '--ctrl_lr',
        type=float,
        default=3.5e-4,
        help='adam: ctrl learning rate')
    parser.add_argument(
        '--lr_decay',
        action='store_true',
        help='learning rate decay or not')
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.0,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.9,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='number of cpu threads to use during batch generation')
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=128,
        help='dimensionality of the latent space')
    parser.add_argument(
        '--img_size',
        type=int,
        default=32,
        help='size of each image dimension')
    
    # Note: also used for trig data generation
    parser.add_argument(
        '--channels',
        type=int,
        default=3,
        help='number of image channels')
    
    
    parser.add_argument(
        '--n_critic',
        type=int,
        default=1,
        help='number of training steps for discriminator per iter')
    parser.add_argument(
        '--val_freq',
        type=int,
        default=20,
        help='interval between each validation')
    parser.add_argument(
        '--print_freq',
        type=int,
        default=100,
        help='interval between each verbose')
    parser.add_argument(
        '--load_path',
        type=str,
        help='The reload model path')
    parser.add_argument(
        '--class_name',
        type=str,
        help='The class name to load in UniMiB dataset')
    parser.add_argument(
        '--augment_times',
        type=int,
        default=None,
        help='The times of augment signals compare to original data')
    parser.add_argument(
        '--exp_name',
        type=str,
        help='The name of exp')
    parser.add_argument(
        '--d_spectral_norm',
        type=str2bool,
        default=False,
        help='add spectral_norm on discriminator?')
    parser.add_argument(
        '--g_spectral_norm',
        type=str2bool,
        default=False,
        help='add spectral_norm on generator?')
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        help='dataset type')
    # currently supports
    # 'UniMiB', 'particle_data', 'sine', 'cosine', 'stock_data'
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data',
        help='The path of data set')
    parser.add_argument('--init_type', type=str, default='normal',
                        choices=['normal', 'orth', 'xavier_uniform', 'false'],
                        help='The init type')
    
    #NOTE: BY AHMAD: following two parameters are being used as embed dim
    parser.add_argument('--gf_dim', type=int, default=64,
                        help='The base channel num of gen')
    parser.add_argument('--df_dim', type=int, default=64,
                        help='The base channel num of disc')
    
    parser.add_argument(
        '--gen_model',
        type=str,
        help='path of gen model')
    parser.add_argument(
        '--dis_model',
        type=str,
        help='path of dis model')
    parser.add_argument(
        '--controller',
        type=str,
        default='controller',
        help='path of controller')
    parser.add_argument('--eval_batch_size', type=int, default=100)
    parser.add_argument('--num_eval_imgs', type=int, default=50000)
    parser.add_argument(
        '--bottom_width',
        type=int,
        default=4,
        help="the base resolution of the GAN")
    parser.add_argument('--random_seed', type=int, default=12345)

    # search
    parser.add_argument('--shared_epoch', type=int, default=15,
                        help='the number of epoch to train the shared gan at each search iteration')
    parser.add_argument('--grow_step1', type=int, default=25,
                        help='which iteration to grow the image size from 8 to 16')
    parser.add_argument('--grow_step2', type=int, default=55,
                        help='which iteration to grow the image size from 16 to 32')
    parser.add_argument('--max_search_iter', type=int, default=90,
                        help='max search iterations of this algorithm')
    parser.add_argument('--ctrl_step', type=int, default=30,
                        help='number of steps to train the controller at each search iteration')
    parser.add_argument('--ctrl_sample_batch', type=int, default=1,
                        help='sample size of controller of each step')
    parser.add_argument('--hid_size', type=int, default=100,
                        help='the size of hidden vector')
    parser.add_argument('--baseline_decay', type=float, default=0.9,
                        help='baseline decay rate in RL')
    parser.add_argument('--rl_num_eval_img', type=int, default=5000,
                        help='number of images to be sampled in order to get the reward')
    parser.add_argument('--num_candidate', type=int, default=10,
                        help='number of candidate architectures to be sampled')
    parser.add_argument('--topk', type=int, default=5,
                        help='preserve topk models architectures after each stage' )
    parser.add_argument('--entropy_coeff', type=float, default=1e-3,
                        help='to encourage the exploration')
    parser.add_argument('--dynamic_reset_threshold', type=float, default=1e-3,
                        help='var threshold')
    parser.add_argument('--dynamic_reset_window', type=int, default=500,
                        help='the window size')
    parser.add_argument('--arch', nargs='+', type=int,
                        help='the vector of a discovered architecture')
    parser.add_argument('--optimizer', type=str, default="adam",
                        help='optimizer')
    parser.add_argument('--loss', type=str, default="hinge",
                        help='loss function')
    parser.add_argument('--n_classes', type=int, default=0,
                        help='classes')
    parser.add_argument('--phi', type=float, default=1,
                        help='wgan-gp phi')
    parser.add_argument('--grow_steps', nargs='+', type=int,
                        help='the vector of a discovered architecture')
    parser.add_argument('--D_downsample', type=str, default="avg",
                        help='downsampling type')
    parser.add_argument('--fade_in', type=float, default=1,
                        help='fade in step')
    parser.add_argument('--d_depth', type=int, default=7,
                        help='Discriminator Depth')
    #parser.add_argument('--g_depth', type=str, default="5,4,2",
    #                    help='Generator Depth')
    parser.add_argument('--g_norm', type=str, default="ln",
                        help='Generator Normalization')
    parser.add_argument('--d_norm', type=str, default="ln",
                        help='Discriminator Normalization')
    parser.add_argument('--g_act', type=str, default="gelu",
                        help='Generator activation Layer')
    parser.add_argument('--d_act', type=str, default="gelu",
                        help='Discriminator activation layer')
    parser.add_argument('--patch_size', type=int, default=4,
                        help='Discriminator Depth')
    parser.add_argument('--fid_stat', type=str, default="None",
                        help='Discriminator Depth')
    parser.add_argument('--diff_aug', type=str, default="None",
                        help='differentiable augmentation type')
    parser.add_argument('--accumulated_times', type=int, default=1,
                        help='gradient accumulation')
    parser.add_argument('--g_accumulated_times', type=int, default=1,
                        help='gradient accumulation')
    parser.add_argument('--num_landmarks', type=int, default=64,
                        help='number of landmarks')
    parser.add_argument('--d_heads', type=int, default=4,
                        help='number of heads')
    #parser.add_argument('--dropout', type=float, default=0.,
    #                    help='dropout ratio')
    parser.add_argument('--ema', type=float, default=0.995,
                        help='ema')
    parser.add_argument('--ema_warmup', type=float, default=0.,
                        help='ema warm up')
    parser.add_argument('--ema_kimg', type=int, default=500,
                        help='ema thousand images')
    parser.add_argument('--latent_norm',action='store_true',
        help='latent vector normalization')
    parser.add_argument('--ministd',action='store_true',
        help='mini batch std')
    parser.add_argument('--g_mlp', type=int, default=4,
                        help='generator mlp ratio')
    parser.add_argument('--d_mlp', type=int, default=4,
                        help='discriminator mlp ratio')
    parser.add_argument('--g_window_size', type=int, default=8,
                        help='generator mlp ratio')
    parser.add_argument('--d_window_size', type=int, default=8,
                        help='discriminator mlp ratio')
    parser.add_argument('--show', action='store_true',
                    help='show')
    
    #NOTE:ADDED BY AHMAD
    parser.add_argument('--seq_len', type=int, default=4,
                        help='sequence length')
    parser.add_argument('--g_heads', type=int, default=4,
                        help='number of heads for generator')
    
    parser.add_argument('--g_depth', type=int, default=5,
                        help='Generator Depth')
    
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout ratio for all layers except the argument "forward_dropout"')
    parser.add_argument('--forward_dropout', type=float, default=0.5,
                        help='dropout ratio for FC layers in encoder of transformer')
    
    # IN TTS_GAN, the dropout is set to 0.5 while in informer is 0.1
    
    parser.add_argument(
        '--is_normalize',
        action='store_true',
        help='whether to normalize dataset')
    parser.add_argument('--norm_type', type=str, default="StandardScaler",
                        help='Type of Normalization (StandardScaler, MinMax).')
    parser.add_argument('--norm_level', type=str, default="sample",
                        help='perform normalization at sample level or dataset level(sample or full).')
   
    
    # settings for particle data
    parser.add_argument('--which_folder_data', type=str, default="complete",
                        help='which folder data to be used during particle data processing.')
    
    parser.add_argument('--fold_no', type=int, default=0,
                        help='Fold number for particle data')
    
    parser.add_argument(
        '--upsample_Y',
        action='store_true',
        help='whether to upsample Y in particle data')
    
    # settings for trigonometric data
    parser.add_argument('--number_of_samples', type=int, default=10000,
                        help='Number of samples for trigonometric data')
    
    
    # settings for informer encoder
    parser.add_argument('--model_type', type=str, required=True, default='transformer',help='model of experiment, options: [transformer, informer, informerstack]')

    parser.add_argument('--g_s_layers', type=str, default='3,2,1', help='num of stack encoder layers for generator')
    parser.add_argument('--d_s_layers', type=str, default='3,2,1', help='num of stack encoder layers for discriminator')
    
    
    #parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
    
    #parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
    
    
    parser.add_argument('--save_model_freq', type=int, default=-1, help='Save model freq')
    parser.add_argument('--plot_syn_data',action='store_true',
        help='whether to plot synthetic data during training. It is created when model is saved (controlled by save_model_freq).Can be memory intensive.')
    
    opt = parser.parse_args()

    return opt