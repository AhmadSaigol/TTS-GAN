"""
Generates and saves synthetic dataset    
"""

import os

import numpy as np
import torch

from GANModels import Generator
from read_parameters_from_log import find_value_from_log
from utils.json_processing import load_from_json

def generate_synthetic_data(path_to_run, sample_size, cp='all', denormalize=False, file_name='syn_dataset'):
    """
    Generates and saves synthetic data.(shape (number_of_samples, seq_len, number_fo_features))

    Args:
        path_to_run (str): path to the run
        sample_size (int): number of samples to generate
        cp(str/int): checkpoint to use ('all' or checkpoint number)
        denormalize: whether to denormalize data
        file_name: name of file
    
    """
    path_to_log_folder = os.path.join(path_to_run, 'Log')

    # read log file
    log_file = [f for f in os.listdir(path_to_log_folder) if '_train.log' in f]
    path_to_log_file = os.path.join(path_to_log_folder, log_file[0])

    with open(path_to_log_file, 'r') as f:
        log_data = f.read()


    # extract parameters from log file
    model_type = find_value_from_log('model_type', log_data)[1:-1]
    seq_len = int(find_value_from_log('seq_len', log_data))
    patch_size = int(find_value_from_log('patch_size', log_data))
    channels = int(find_value_from_log('channels', log_data))
    latent_dim = int(find_value_from_log('latent_dim', log_data))
    gf_dim = int(find_value_from_log('gf_dim', log_data))
    g_depth = int(find_value_from_log('g_depth', log_data))
    d_depth = int(find_value_from_log('g_depth', log_data))
    g_heads = int(find_value_from_log('g_heads', log_data))
    forward_dropout = float(find_value_from_log('forward_dropout', log_data))
    dropout = float(find_value_from_log('dropout', log_data))
    factor = int(find_value_from_log('factor', log_data))
    attn = find_value_from_log('attn', log_data)[1:-1]
    #TODO
    g_s_layers = find_value_from_log('g_s_layers', log_data)
    d_s_layers = find_value_from_log('d_s_layers', log_data)
  
  
    # set up normalization
    if denormalize:
        is_normalize = bool(find_value_from_log('is_normalize'))
        norm_level = find_value_from_log('norm_level')[1:-1]
        norm_type = find_value_from_log('norm_type')[1:-1]
        assert norm_level in ['sample', 'full']
        assert norm_type in ['MinMax', 'StandardScaler']
        
        if not is_normalize:
            raise ValueError(f"No normalization was set for the run:{path_to_run}")
        if norm_level == 'sample':
             raise ValueError(f"The de-normalization is not supported for {norm_level}.")
          
        # load norm values
        path_to_norm_values = os.path.join(path_to_run, 'Data', 'norm_values')
        norm_values_dict = load_from_json(path_to_norm_values)
            
    if model_type == 'transformer' or model_type=='informer':
        g_depth = g_depth
        d_depth = d_depth
    elif model_type=='informerstack':
        g_depth = [int(s_l) for s_l in g_s_layers.replace(' ','').split(',')]
        d_depth = [int(s_l) for s_l in d_s_layers.replace(' ','').split(',')]
    else:
       raise ValueError(f"Unknown type of model type encountered: {model_type}") 
   
    # set up model
    gen_net = Generator(seq_len=seq_len,
                        patch_size=patch_size, 
                        channels=channels, 
                        latent_dim=latent_dim, 
                        embed_dim=gf_dim, 
                        depth=g_depth,
                        num_heads=g_heads,
                        forward_drop_rate=forward_dropout, 
                        attn_drop_rate=dropout, 
                        
                        model_type=model_type,
                        
                        factor= factor,
                        attn=attn,
                         
                        ).cuda()
    gen_net.eval()

    # get checkpoints
    path_to_model = os.path.join(path_to_run, 'Model')
    if cp == 'all':
        checkpoints = os.listdir(path_to_model)
    else:
        checkpoints = [f'checkpoint_{cp}']
    
    # set up directory for results 
    path_to_results = os.path.join(path_to_run, 'Synthetic_data')
    if not os.path.exists(path_to_results):
        os.mkdir(path_to_results)
        
    for cpkt in checkpoints:
        
        print(f"Processing : {cpkt}")
        path_to_checkpoint = os.path.join(path_to_model, cpkt)#TODOprint
    
        curr_cpkt_model = torch.load(path_to_checkpoint)
        _ = gen_net.load_state_dict(curr_cpkt_model['gen_state_dict'])
        print(_)

        # Sample noise as generator input (TODO: same Z for all checkpoints or different for each checkpoint)
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (sample_size, latent_dim))).cuda()

        # generate data
        syn_data = gen_net(z)
        syn_data = syn_data.detach().cpu().numpy()

        # reshape data
        syn_data = np.transpose(syn_data[:,:,0,:], (0,2,1))
        
        if denormalize:
            print(f"Denormalizing data . . . norm type: {norm_type} ")
            if norm_type == 'MinMax':
                min_values = np.array(norm_values_dict['min'])
                max_values = np.array(norm_values_dict['max'])
                syn_data = syn_data *(max_values-min_values) + min_values
                
            if norm_type == 'StandardScaler':
                mean_values = np.array(norm_values_dict['mean'])
                std_values = np.array(norm_values_dict['std'])
                
                syn_data = syn_data *std_values + mean_values
                
        
        # save data
        path_to_curr_synthetic_data = os.path.join(path_to_results, cpkt)
        if not os.path.exists(path_to_curr_synthetic_data):
            os.mkdir(path_to_curr_synthetic_data)
        np.save(os.path.join(path_to_curr_synthetic_data, f'{file_name}.npy'), syn_data)

if __name__ == '__main__':

    number_of_runs = 1
    
    path_to_folder = "logs"
    run_list = ['Full_STOCKS_DATA_2025_04_04_23_08_00']
    
    sample_size_list = [
        3662
    ]
    denormalize_list = [      
        False
    ]*number_of_runs
    
    for run_no in range(number_of_runs):
        print(f"Prcoessing run no: {run_no} run: {run_list[run_no]}")
        
        path_to_run = os.path.join(path_to_folder, run_list[run_no])
        
        denormalize= denormalize_list[run_no]
        sample_size =sample_size_list[run_no]
        
        generate_synthetic_data(path_to_run, sample_size, denormalize=denormalize)
    print("Processing Completed.")
