import numpy as np
import torch
from GANModels import Discriminator, Generator
from read_parameters_from_log import find_value_from_log
import os
import time
import pickle
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

import torch.nn as nn
from utils.json_processing import save_to_json, load_from_json


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap



# uncomment when running code not in graph mode (False)
#tf.config.run_functions_eagerly(True)


# NOTES:
  # NOT SET UP FOR VARIABLE LENGTH SNIPPETS
  # PLOTS WEIGHT INDIVIDUALLY OVER TIME (TOO MANY) OR Hinton Diagrams OVER TIME IN VIDEO WITH REGIONS MARKING LAYERS AND MODELS OR SOMETHING ELSE 
  # PLOT MODEL INPUT/OUTPUT (TOO MANY)



def create_plots(cp, E_loss0, E_loss, D_loss, G_loss, G_loss_S, X, X_hat, i):
    sample_size = 1000
    idx = np.random.permutation(X.shape[0])[:sample_size]

    X = X[idx]
    X_hat = X_hat[idx]

    X_mean = np.mean(X, 2)
    X_hat_mean = np.mean(X_hat, 2)

    # TSNE
    data = np.concatenate((X_mean, X_hat_mean), axis = 0)

    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(data)

    # PCA
    pca = PCA(n_components = 2)
    pca.fit(X_mean)
    pca_results = pca.transform(X_mean)
    pca_hat_results = pca.transform(X_hat_mean)


    if not os.path.exists(PATH_TO_RESULTS[i]):
        os.mkdir(PATH_TO_RESULTS[i])
    
    fig = plt.figure(figsize=(10,10), layout='constrained')
    
    #gs = fig.add_gridspec(6, 6,  width_ratios=(1, 2), height_ratios=(1, 4),
    #                  left=0.1, right=0.9, bottom=0.1, top=0.9,
    #                  wspace=0.05, hspace=0.05)
    gs = GridSpec(6, 6, figure=fig)

    # feature axes
    ax5 = fig.add_subplot(gs[5, 0:2])

    ax0 = fig.add_subplot(gs[0, 0:2], sharex=ax5)
    ax1 = fig.add_subplot(gs[1, 0:2], sharex=ax5)
    ax2 = fig.add_subplot(gs[2, 0:2], sharex=ax5)
    ax3 = fig.add_subplot(gs[3, 0:2], sharex=ax5)
    ax4 = fig.add_subplot(gs[4, 0:2], sharex=ax5)
    
    
    # PCA  and tsne axes
    ax6 = fig.add_subplot(gs[0:3, 2:])
    ax7 = fig.add_subplot(gs[3:, 2:])

    fig.suptitle(f'cp: {cp} E_loss0: {E_loss0:.4f} E_loss: {E_loss:.4f} D_loss: {D_loss:.4f} G_loss: {G_loss:.4f} G_loss_S: {G_loss_S:.4f}'.format(E_loss0, E_loss, D_loss, G_loss, G_loss_S))
    
    # plot data
    ax0.set_title('Comparsion b/w Original and Sythetic')
    ax0.plot(X[0, :, 0], c='red')
    ax0.plot(X_hat[0, :, 0], c='blue')
    ax0.set_ylabel(f'feature_0')
    ax0.set_xticks([])
    
    ax1.plot(X[0, :, 1], c='red')
    ax1.plot(X_hat[0, :, 1], c='blue')
    ax1.set_ylabel(f'feature_1')
    ax1.set_xticks([])

    ax2.plot(X[0, :, 2], c='red')
    ax2.plot(X_hat[0, :, 2], c='blue')
    ax2.set_ylabel(f'feature_2')
    ax2.set_xticks([])
    
    ax3.plot(X[0, :, 3], c='red')
    ax3.plot(X_hat[0, :, 3], c='blue')
    ax3.set_ylabel(f'feature_3')
    ax3.set_xticks([])

    ax4.plot(X[0, :, 4], c='red')
    ax4.plot(X_hat[0, :, 4], c='blue')
    ax4.set_ylabel(f'feature_4')
    ax4.set_xticks([])

    ax5.plot(X[0, :, 5], c='red')
    ax5.plot(X_hat[0, :, 5], c='blue')
    ax5.set_ylabel(f'feature_5')
    ax5.set_xlabel(f'timestep')

    # plot pca
    ax6.scatter(pca_results[:,0], pca_results[:,1], c='red', alpha=0.2, label='Original')
    ax6.scatter(pca_hat_results[:,0], pca_hat_results[:,1], c='blue', alpha=0.2, label='Synthetic')
    ax6.legend()
    ax6.set_title('PCA')
    ax6.set_xlabel('component-1')
    ax6.set_ylabel('component-2')
    
    # plot tsne
    ax7.scatter(tsne_results[:X.shape[0],0], tsne_results[:X.shape[0],1], c='red', alpha=0.2, label='Original')
    ax7.scatter(tsne_results[X.shape[0]:,0], tsne_results[X.shape[0]:,1], c='blue', alpha=0.2, label='Synthetic')
    ax7.legend()
    ax7.set_title('tSNE')
    ax7.set_xlabel('component-1')
    ax7.set_ylabel('component-2')
    
    plt.savefig(os.path.join(PATH_TO_RESULTS[i], f'plot_{cp}.png'))
    plt.close()

def find_cp(cps):
    """
    finds last checkpoint in the list
    :param: list containing checkpoints
    :return: last checkpoint
    """
    
    if len(cps)==1:
        return cps[0]
    else:
        max_cp = 0
        max_cp_id = '0'
        for cp in cps:
            cp_num = int(cp)
            if cp_num > max_cp:
                max_cp = cp_num
                max_cp_id = cp
    return max_cp_id 

def sort_cps(cps):
        """
        Sort cps ids w.r.t cp numbers
        :param: list containing cps
        :return: sorted list containing cps
        """
        cps_less_than_10 = []
        cps_less_than_100 = []
        cps_less_than_1000 = []
        cps_less_than_10000 = []
        remaining_cps = []
        
        for cp in cps:
          
            cp_num = int(cp)
            if cp_num < 10:
                cps_less_than_10.append(cp)
            elif cp_num < 100:
                cps_less_than_100.append(cp)
            elif cp_num < 1000:
                cps_less_than_1000.append(cp)
            elif cp_num < 10000:
                cps_less_than_10000.append(cp)
            else:
                remaining_cps.append(cp)
        
        cps_less_than_10.sort()
        cps_less_than_100.sort()
        cps_less_than_1000.sort()
        cps_less_than_10000.sort()

        _cps = cps_less_than_10 + cps_less_than_100 + cps_less_than_1000 + cps_less_than_10000

        return _cps

def create_labels_and_vertical_lines(save_model_freq,
                              embedding_network_iters,
                              supervisor_network_iters,
                              discriminator_network_iters,
                              generator_network_iters):
    """
    Maps checkpoint no to number of iterations during training
    Also, Finds locations (number of iteration) where each phase ends during the training

    Parameters:
        save_model_freq: the freq at which model was saved during the training
        embedding_network_iters: number of iterations for training embedder 
        supervisor_network_iters: number of iterations for training supervisor 
        discriminator_network_iters: number of iterations for training discriminator 
        generator_network_iters: number of iterations for training generator
    
    Returns:
      labels: list containing mapped checkpoints (number of iteration)
      vl: list containing mapped checkpoints (number of iteration) where each phase ends
    
    Note:
      - If the way of saving model during training is changed, the code has no way to correct itself.
      - Currently, it assumes the following:
          - model before the start of the training is saved.
          - during the training of each phase, saving of model is controlled by respective iterations. 
          - No model is saved during generator_network_iters.
          - At the end of each phase of training, model is saved.
      - The model saved at the last iteration (if saved) and the model at the end of phase would represent the same model.
        These models are differentiated from each other in return labels by small offset(0.5)
    
    """    
    if save_model_freq >1:    
      labels = [0]
      vl = []

      # autoencoder training
      '''
      for it in range(embedding_network_iters):
          if it % save_model_freq ==0:
              if it == 0:
                  labels.append(1)
              else:
                  labels.append(it)
      '''
      temp = [it for it in range(embedding_network_iters) if it % save_model_freq ==0]
      temp[0] =1
      
      labels += temp

      # add small offset if last iter model was saved twice
      if labels[-1] != embedding_network_iters-1:
          labels.append(embedding_network_iters)
      else:
          labels.append(embedding_network_iters + 0.5)
      
      vl.append(labels[-1])
      offset = embedding_network_iters #+1 
      
      #supervisor training
      '''
      for it in range(supervisor_network_iters):
          if it % save_model_freq ==0:
              labels.append(offset+it)
      '''
      temp = [offset+it for it in range(supervisor_network_iters) if it % save_model_freq ==0]
      temp[0]+=1
      labels += temp

      # add small offset if last iter model was saved twice
      if labels[-1] != offset + supervisor_network_iters-1:
          labels.append(offset + supervisor_network_iters)
      else:
          labels.append(offset + supervisor_network_iters + 0.5)

      vl.append(labels[-1])
      offset = offset + supervisor_network_iters #+ 1

      # joint training
      '''
      for it in range(discriminator_network_iters):
          if it % save_model_freq ==0:
              labels.append(offset+it)
      '''
      temp = [offset+it for it in range(discriminator_network_iters) if it % save_model_freq ==0]
      temp[0]+=1
      labels += temp

      # add small offset if last iter model was saved twice
      if labels[-1] != offset + discriminator_network_iters-1:
          labels.append(offset + discriminator_network_iters)
      else:
          labels.append(offset + discriminator_network_iters + 0.5)

      vl.append(labels[-1])
    
    else:
      labels =[0]
      vl = [] 
      temp_fnt = lambda iters, offset: [i + offset for i in range(iters)]
      
      # autoencoder
      ofset =1 
      labels+= temp_fnt(embedding_network_iters, ofset)
      labels.append(embedding_network_iters + 0.5)
      vl.append(labels[-1])

      # supervisor
      ofset += embedding_network_iters
      labels+= temp_fnt(supervisor_network_iters, ofset)
      labels.append(ofset + supervisor_network_iters-1 + 0.5)
      vl.append(labels[-1])

      # discriminator
      ofset += supervisor_network_iters 
      labels+= temp_fnt(discriminator_network_iters, ofset)
      labels.append(ofset + discriminator_network_iters -1 + 0.5)
      vl.append(labels[-1])


    return labels, vl

def create_training_animation_for_loss_pca_tsne(generator_loss, 
                              discriminator_loss,
                              tsne_results, 
                              pca_results,  
                              fps, 
                              xlabels,
                              path_to_results,
                              **kwargs):
    """
      Creates and saves an animation depicting how loss, pca and tsne evolve during the training  
    
      Parameters:
          generator_loss: list containing autoencoder loss for all cps 
          discriminator_loss: list containing discriminator loss for all cps
          tsne_results: list containing tsne results for all cps
          pca_results: list containing pca results for all cps
          xlabels:
          fps: frames per second for the animation
          path_to_results: path to the folder where results will be saved
      
      Keywords:
        file_name: suffix to add to file name
        interval: Delay between frames in milliseconds (default=3)    
    """
    suffix = kwargs.get('file_name', '')
    file_name = 'loss_pca_tsne_training_animation'
    if suffix:
      file_name += f'_{suffix}'
    file_name+='.mp4'

    interval = kwargs.get('interval', 3)

    sample_size = pca_results[0].shape[0]//2

    min_x_value = np.min(xlabels)
    max_x_value = np.max(xlabels)
    
    number_of_cps = len(generator_loss)
    #cps = [i for i in range(number_of_cps)]

    if number_of_cps != len(xlabels):
        raise ValueError(f"Number of checkpoints({number_of_cps}) do not match with the number of created labels({len(xlabels)})")

    fig = plt.figure(figsize=(13,13), layout='constrained')
    gs = GridSpec(10, 10, figure=fig)

    # loss axes
    ax1 = fig.add_subplot(gs[5:, 0:4])
    ax0 = fig.add_subplot(gs[0:5, 0:4], sharex=ax1)
  
    # PCA and tSNE axes
    ax5 = fig.add_subplot(gs[0:5, 4:])
    ax6 = fig.add_subplot(gs[5:, 4:])

    title = fig.suptitle(f'Training process visualization - Epoch {xlabels[0]}')

    # plot generator_loss
    generator_loss_min = np.min(generator_loss)
    generator_loss_max = np.max(generator_loss)
    #autoencoder_line = ax0.plot(autoencoder_loss[0], cps[0])[0]
    generator_line = ax0.plot(generator_loss[0], xlabels[0])[0]
    generator_text = ax0.set_title('Generator Loss: %.4f' % generator_loss[0])
    #ax0.set_xticks([])
    ax0.set_ylim([generator_loss_min-0.1*generator_loss_min, generator_loss_max+0.1*generator_loss_max])
    #ax0.set_xlim([0,number_of_cps-1])
    ax0.set_xlim([min_x_value, max_x_value])
    
    # plot discriminator loss
    discriminator_loss_min = np.min(discriminator_loss)
    discriminator_loss_max = np.max(discriminator_loss)
    #discrim_line = ax4.plot(discrim_loss[0], cps[0])[0]
    #discrim_line = ax4.plot(discrim_loss[0], cps[0])[0]
    discriminator_line = ax1.plot(discriminator_loss[0], xlabels[0])[0]
    discriminator_text = ax1.set_title('Discriminator Loss: %.4f' % discriminator_loss[0])
    ax1.set_xlabel(f'Epoch')
    ax1.set_ylim([discriminator_loss_min-0.1*discriminator_loss_min, discriminator_loss_max+0.1*discriminator_loss_max])
    #ax4.set_xlim([0,number_of_cps-1])
    ax1.set_xlim([min_x_value, max_x_value])

    #ax4.set_xticklabels(labels)
    
    '''
    # find min and max values in pca results
    pca_x_min = []
    pca_x_max = []
    pca_y_min = []
    pca_y_max = []

    for i in range(number_of_cps):
        pca_x_min.append(np.min(pca_results[i][:,0]))
        pca_x_min.append(np.min(pca_hat_results[i][:,0]))
        pca_x_max.append(np.max(pca_results[i][:,0]))
        pca_x_max.append(np.max(pca_hat_results[i][:,0]))

        pca_y_min.append(np.min(pca_results[i][:,1]))
        pca_y_min.append(np.min(pca_hat_results[i][:,1]))
        pca_y_max.append(np.max(pca_results[i][:,1]))
        pca_y_max.append(np.max(pca_hat_results[i][:,1]))
    
    pca_x_min = np.min(pca_x_min)
    pca_x_max = np.max(pca_x_max)
    pca_y_min = np.min(pca_y_min)
    pca_y_max = np.max(pca_y_max)
    '''

    # plot pca
    #pca_x = np.concatenate((pca_results[0][:,0], pca_hat_results[0][:,0]), axis=0)
    #pca_y = np.concatenate((pca_results[0][:,1], pca_hat_results[0][:,1]), axis=0)
    c = [0]*sample_size + [1]*sample_size 
    color_map =  ListedColormap(['r', 'b'])
    names = ['Original', 'Synthetic']
    #ax6.scatter(pca_results[0][:,0], pca_results[0][:,1], c='red', alpha=0.2, label='Original')
    pca_scatter= ax5.scatter(pca_results[0][:,0], pca_results[0][:,1], c=c, alpha=0.2, cmap=color_map)
    ax5.legend(handles=pca_scatter.legend_elements()[0], labels=names, loc='upper right')
    #ax5.set_ylim([pca_y_min,pca_y_max])
    #ax5.set_xlim([pca_x_min,pca_x_max])
    ax5.set_title('PCA')
    ax5.set_xlabel('component-1')
    ax5.set_ylabel('component-2')

    # find min and max values in tsne results
    '''
    tsne_x_min = []
    tsne_x_max = []
    tsne_y_min = []
    tsne_y_max = []

    for i in range(number_of_cps):
        tsne_x_min.append(np.min(tsne_results[i][:,0]))
        tsne_x_max.append(np.max(tsne_results[i][:,0]))
        
        tsne_y_min.append(np.min(tsne_results[i][:,1]))
        tsne_y_max.append(np.max(tsne_results[i][:,1]))
        
    tsne_x_min = np.min(tsne_x_min)
    tsne_x_max = np.max(tsne_x_max)
    tsne_y_min = np.min(tsne_y_min)
    tsne_y_max = np.max(tsne_y_max)
    '''
    
    # plot tsne
    tsne_scatter=ax6.scatter(tsne_results[0][:,0], tsne_results[0][:,1], c=c, alpha=0.2, cmap=color_map)
    #ax6.scatter(tsne_results[X.shape[0]:,0], tsne_results[X.shape[0]:,1], c='blue', alpha=0.2, label='Synthetic')
    ax6.legend(handles=tsne_scatter.legend_elements()[0], labels=names, loc='upper right')
    #ax5.set_ylim([tsne_y_min,tsne_y_max])
    #ax5.set_xlim([tsne_x_min,tsne_x_max])
    ax6.set_title('tSNE')
    ax6.set_xlabel('component-1')
    ax6.set_ylabel('component-2')

    def update(frame):
        """
        Updates the values in the frame of the animation
        Parameter:
          frame: int
        """
        
        #autoencoder_line.set_xdata(cps[:frame])
        generator_line.set_xdata(xlabels[:frame])
        generator_line.set_ydata(generator_loss[:frame])
        generator_text.set_text('Generator Loss: %.4f' % generator_loss[frame])
        
        #discrim_line.set_xdata(cps[:frame])
        discriminator_line.set_xdata(xlabels[:frame])
        discriminator_line.set_ydata(discriminator_loss[:frame])
        discriminator_text.set_text('Discriminator Loss: %.4f' % discriminator_loss[frame])

        #pca_x = np.concatenate((pca_results[frame][:,0], pca_hat_results[frame][:,0]), axis=0)
        #pca_y = np.concatenate((pca_results[frame][:,1], pca_hat_results[frame][:,1]), axis=0)
        
        pca_data = np.stack([pca_results[frame][:,0], pca_results[frame][:,1]]).T
        pca_scatter.set_offsets(pca_data)

        tsne_data = np.stack([tsne_results[frame][:,0], tsne_results[frame][:,1]]).T
        tsne_scatter.set_offsets(tsne_data)
        
        title.set_text(f'Training process visualization - Epoch {xlabels[frame]}')
        
        return (generator_line, discriminator_line, pca_scatter, tsne_scatter, generator_text, discriminator_text, title)
         
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=number_of_cps, interval=interval)
    ani.save(filename=os.path.join(path_to_results, file_name), writer="ffmpeg", fps=fps)

    plt.close()
    
def find_num_rows_and_cols_for_figure(num_features):
    """
    Finds the number of rows and cols for the figure
    """
    temp_rows = np.sqrt(num_features)
    num_rows = 0
    num_cols = 0
    diff = 0

    for i in range(2):
      if i==0:
        r = np.floor(temp_rows)
      else:
        r = np.ceil(temp_rows)
      
      for j in range(2):
        
        if j == 0:
          c = np.floor(num_features/r)
        else:
          c = np.ceil(num_features/r)

        # figure not big enough to have all features
        if r*c < num_features:
          continue
        else:
          curr_diff = np.abs(r-c)
          if diff ==0: #first time
            num_rows = r
            num_cols = c
            diff = curr_diff
          else: # r and c must be close to each other
            if curr_diff < diff:
              if r < c:
                num_rows = r
                num_cols = c
              else:
                num_rows = c
                num_cols = r

    if num_cols ==0 or num_rows ==0:
      raise ValueError("Something went wrong while calculating number of rows and cols.")
    else:
      return int(num_rows), int(num_cols)

def create_training_animation_for_closet_sample(syn_data, ori_data,
                              fps, 
                              path_to_results,
                              **kwargs):
    """
      Creates and saves an animation depicting how a generated sample evolves during the training process  
    
      Parameters:
          syn_data: numpy array of shape(num_cps, num_ts, num_fs)
          ori_data: numpy array of shape(num_ts, num_fs)
          fps: frames per second for the animation
          path_to_results: path to the folder where results will be saved
      
      Keywords:
        file_name: suffix to add to file name    
        normalization_values: a dict containing normalization values. (default={} -> no denormalization is performed)
              keys: 
                type: min_max -> keys: min and max
                type: standard_scaler keys: mean and std
              
        feature_names: list containing feature names
        interval: Delay between frames in milliseconds (default=3)
        epochs:
    """
    # set up file name
    suffix = kwargs.get('file_name', '')
    file_name = 'closest_sample_training_animation'
    if suffix:
      file_name += f'_{suffix}'
    file_name+='.mp4'

    # denormalize data
    normalization_values = kwargs.get('normalization_values', {})
    if normalization_values:
      if normalization_values['type'] == 'min_max':   
          min_vals = np.array(normalization_values['min'])
          max_vals = np.array(normalization_values['max'])
          syn_data = syn_data*(max_vals - min_vals) + min_vals
          ori_data = ori_data*(max_vals - min_vals) + min_vals
      
      if normalization_values['type'] == 'standard_scaler':
          mean_vals = np.array(normalization_values['mean'])
          std_vals =  np.array(normalization_values['std']) 
          syn_data = syn_data*std_vals + mean_vals
          ori_data = ori_data*std_vals + mean_vals
      
    num_cps, num_ts, num_fs = syn_data.shape 

    # get feature names
    feature_names = kwargs.get('feature_names', [])
    if not feature_names:
       feature_names = [f'f{i}' for i in range(num_fs)]
    
    if len(feature_names) != ori_data.shape[-1] != num_fs:
       raise ValueError("The length of feature_names must be equal to the number of features in original and synthetic data.")

    interval = kwargs.get('interval', 3)
    '''
    # set up figure
    if num_fs >= 40:
      num_rows = 10
      figsize=(15,15)
    else :
      num_rows = 3
      figsize=(10,10)
    num_cols = int(np.ceil(num_fs/num_rows))
    '''
    epochs = kwargs.get('epochs', [i for i in range(num_cps)])
    num_rows, num_cols =find_num_rows_and_cols_for_figure(num_fs)
    figsize = (25,15)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, dpi=300)
    title = fig.suptitle('Training of a Sample-Original(red) and Synthetic(blue)-{suffix}-Epoch {epochs[0]}')

    xlabels = np.linspace(0, num_ts-1, 5)
    
    row =0
    col =0

    # get axes limits
    min_fs_values = np.min(np.concatenate((syn_data, ori_data[None, :]), axis=0), axis=(0,1))
    max_fs_values = np.max(np.concatenate((syn_data, ori_data[None, :]), axis=0), axis=(0,1))

    axes_lines = []
    for i in range(num_rows*num_cols):
      
      if i < num_fs:
        # plot original and synthetic data
        axes[row, col].plot(ori_data[:, i],  c='red')
        curr_ax_line = axes[row, col].plot(syn_data[0, :, i],  c='blue')
        
        axes_lines.append(curr_ax_line)

        # NOTE: this may need to go: if diff between start and end too large, lines not visible
        axes[row, col].set_ylim([min_fs_values[i]-0.1*min_fs_values[i], max_fs_values[i]+0.1*max_fs_values[i]])
        
        #axes[row, col].set_ylabel(self.feature_names[i])
        axes[row, col].text(0.1, 0.95, feature_names[i], horizontalalignment='left', verticalalignment='top', transform=axes[row, col].transAxes)
        #axes[row, col].legend()
        axes[row, col].set_xticks([])
      
      else:
        # remove extra subplots
        fig.delaxes(axes[row, col])

      if i== num_fs-1:
          axes[row, col].set_xlabel(f'timestep')
          axes[row, col].set_xticks(xlabels, labels=xlabels)

      if row == num_rows -1:
        if i< num_fs:
          axes[row, col].set_xlabel(f'timestep')
          axes[row, col].set_xticks(xlabels, labels=xlabels)
        col+=1
        row=0
      else:
        row+=1 

    def update(frame):
        """
        Updates the values in the frame of the animation
        Parameter:
          frame: int
        """
        for i in range(num_fs):
           axes_lines[i][0].set_ydata(syn_data[frame, :, i])
        
        title.set_text(f'Training of a Sample-Original(red) and Synthetic(blue)-{suffix}-Epoch {epochs[frame]}')
        
        return (axes_lines, title)
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=num_cps, interval=interval)
    ani.save(filename=os.path.join(path_to_results, file_name), writer="ffmpeg", fps=fps)

    plt.close()

def plot_data(data, title, path_to_results):
    """
    Plots a graph between feature and timesteps(in data).

    Parameters:
      data: can be of shape (num_timesteps, num_features) or (num_features). In the latter case, a constant line is plotted.
      title: title of the figure and file name
    
    """
    if len(data.shape) == 2:
      pass
    elif len(data.shape) == 1:
      data = np.repeat(data[None,:], 10, axis=0) # helps in creating a constant line for each feature
    else:
      raise ValueError(f"Unknown shape of data encountered while ploting. Title: {title} Shape:{data.shape}")
   
    num_timesteps, num_features = data.shape
    
    # plot when there is a single feature
    if num_features == 1:
      plt.plot(data)
      plt.ylabel('feature_0')
      plt.xlabel('timestep')
      plt.title(title)
    else:
      # set up number of subplots in the row
      num_rows = 6
      temp = num_features /num_rows
      if temp < 1:
        num_rows = num_features
      
      # set up number of subplots in the col. 
      num_columns = int(np.ceil(num_features /num_rows))
      
      fig, axes = plt.subplots(num_rows, num_columns, figsize=(10,10))
      fig.suptitle(title)

      row_no = 0
      col_no = 0
      for i in range(num_features):
        if num_columns <= 1:
          axes[row_no].plot(data[:,i], label=f'f{i}')
          axes[row_no].set_xticks([])
          axes[row_no].legend()
        else:  
          axes[row_no, col_no].plot(data[:,i], label=f'f{i}')
          axes[row_no, col_no].set_xticks([])
          axes[row_no, col_no].legend()
        
        row_no +=1
        
        if row_no == num_rows:
          if num_columns <=1:
            axes[row_no-1].set_xlabel(f'timestep')    
            axes[row_no-1].set_xticks(np.arange(0, num_timesteps, 6))
          else:
            axes[row_no-1, col_no].set_xlabel(f'timestep')
            axes[row_no-1, col_no].set_xticks(np.arange(0, num_timesteps, 6))    
          row_no =0
          col_no+=1
      
      if row_no != 0:
        if num_columns <=1:
          axes[row_no].set_xlabel(f'timestep')
          axes[row_no].set_xticks(np.arange(0, data.shape[0], 6))
        else:
          axes[row_no, col_no].set_xlabel(f'timestep')
          axes[row_no, col_no].set_xticks(np.arange(0, data.shape[0], 6))
        
      fig.tight_layout()
    
    plt.savefig(os.path.join(path_to_results, title+'.png'))
    plt.close()

def plot_models_io(title='test', initialize_random=False, pass_via_model=False, X=None, Z=None):
    """
    Creates plot of inputs and outputs of models 
    
    Parameters:
      title: string with before/after training
      initialize_random: whether to initialize inputs randomly or not. Otherwise they are initialized linearly between 0 and 1.
      pass_via_model: X and Z are passed through the model for plotting input and outputs
      X: input to all models when pass_via_model is False. Optional. Required when pass_via_model is True (shape: (1, timesteps, features))
      Z: input to generator model when pass_via_model is True. (shape: (1, timesteps, features))
      
    """
    print("Ploting Input and Output of Models...")
    
    # generate models if it doesnot exist already
    if not self.models:
      self.get_models()
    
    #-----------------------Plot inputs and outputs using X, Z by passing them through the model-----------------
    if pass_via_model:

      if type(X).__name__ =='NoneType':
          raise ValueError("'X' must be provided while plotting inputs and outputs through 'pass_via_model'.")
      
      if self.model_type == 'TimeGAN':
        # set up timeGAN model
        for m in self.models:
            if m.name == 'embedder':
                embedder_model = m
            elif m.name == 'recovery':
                recovery_model = m
            elif m.name == 'generator':
                generator_model = m
            elif m.name == 'supervisor':
                supervisor_model = m
            elif m.name == 'discriminator':
                discriminator_model = m
            else:
                raise ValueError('Unknown name of model encountered while setting up TimeGAN model')
        
        # plot input of model and generator
        if type(Z).__name__ =='NoneType':
          raise ValueError("'Z' must be provided while plotting inputs and outputs through 'pass_via_model'.")
        else: 
          plot_data(X[0],  'via_model_input_'+title)
          plot_data(Z[0],  'via_model_generator_input_'+title)
          
        # plot output of embedder model
        H = embedder_model(X)
        plot_data(H[0], 'via_model_embedder_output_'+title)
        
        # plot output of recovery model
        X_tilde = recovery_model(H)
        plot_data(X_tilde[0], 'via_model_recovery_output_'+title)

        # plot output of generator model
        E_hat = generator_model(Z)
        plot_data(E_hat[0], 'via_model_generator_output_'+title)

        # plot output of supervisor model having embedder output as input 
        H_hat_supervise = supervisor_model(H)
        plot_data(H_hat_supervise[0], 'via_model_embedder_supervisor_output_'+title)

        # plot output of supervisor model having generator output as input 
        H_hat = supervisor_model(E_hat)
        plot_data(H_hat[0], 'via_model_generator_supervisor_output_'+title)
        
        # plot output of discriminator model having data as input flowing via generator and supervisor output 
        Y_fake = discriminator_model(H_hat)
        plot_data(Y_fake[0], 'via_model_generator_supervisor_discriminator_output_'+title)

        # plot output of discriminator model having generator output as input
        Y_fake_e = discriminator_model(E_hat)
        plot_data(Y_fake_e[0], 'via_model_generator_discriminator_output_'+title)

        # plot output of discriminator model having embedder output as input
        Y_real = discriminator_model(H)
        plot_data(Y_real[0], 'via_model_embedder_discriminator_output_'+title)

        # plot output of recovery model having data as input flowing via generator and supervisor output 
        X_hat = recovery_model(H_hat)
        plot_data(X_hat[0], 'via_model_generator_supervisor_recovery_output(predicted)_'+title)

      elif self.model_type== 'discriminative':
        # plot input
        plot_data(X[0],  'via_model_input_'+title)
        discriminative_model = self.models[0]
        output = discriminative_model(X)
        # plot output
        plot_data(output[0], 'via_model_discrimative_output_'+title)

      elif self.model_type == 'predictive':
        # plot input
        plot_data(X[0],  'via_model_input_'+title)
        predictive_model = self.models[0]
        output = predictive_model(X)
        # plot output
        plot_data(output[0], 'via_model_predictive_output_'+title)

      else:
        raise ValueError("Unknown model type encountered while plotting IO via models.")
    
    #-----------------------Plot input and output by passing input independently to each model-------------------
    else:
      for model in self.models:
        num_timesteps, num_features = model.input_shape[1:]
        
        # set up input
        if type(X).__name__ =='NoneType':
          if initialize_random:
            input = np.random.uniform(0, 1, (num_timesteps, 1))
          else:
            input = np.arange(0, 1, 1/num_timesteps).reshape(-1,1)
          
          input = np.repeat(input, num_features, axis=1)
        elif type(X).__name__ == 'ndarray':
          input = X[0]
        else:
          raise ValueError("unknown value encounted for X")
        
        # plot input
        plot_data(input, model.name + '_input_'+title)
        
        output = model(input[None, :, :])
        # plot output
        plot_data(output[0], model.name + '_output_'+title)

def plot_weights(path=None):
    """
    Plot how weights of the models evolve over time

    path:If provided, path to the folder containing the models with structure 
      folder -> model_name1 -> checkpoint_num -> checkpoint files
                model_name2 ->same as above
    Otherwise, loads model using dir in path_to_results/checkpoints
    """
    print("Plotting weights . . . ")
    if path:
      path_to_folder = path
    else:
      path_to_folder = self.path_to_checkpoints

    # set up model
    if self.model_type == 'TimeGAN':
      models = self.__create_timeGAN_models(plot=True)
    elif self.model_type == 'predictive':
      models = self.__create_predictive_model(plot=True)
    elif self.model_type == 'discriminative':
      models = self.__create_discriminative_model(plot=True)

    self.vertical_lines = np.array(self.vertical_lines)
    model_no = 0

    for model in models:
      path_to_model = os.path.join(path_to_folder, model.name)
      
      # get checkpoints
      cps = os.listdir(path_to_model)
      cps = self.__sort_cps(cps)

      layer_weights={}
      
      for cp in cps:
        path_to_cp = os.path.join(path_to_model, cp)
        # load weights
        model.load_weights(path_to_cp+'/')
        
        number_of_layers = len(model.layers)

        # seaprate weights of each layer in the dict
        for layer_no in range(number_of_layers):
          layer_name = model.layers[layer_no].name
          weights= model.layers[layer_no].get_weights() #get layer weights

          if f'{layer_no}' not in layer_weights.keys():
            layer_weights[f'{layer_no}'] = {}
          
          if 'name' not in layer_weights[f'{layer_no}'].keys():
            if 'dense' in layer_name:
              layer_weights[f'{layer_no}']['name'] = 'dense'
              layer_weights[f'{layer_no}']['weights'] = weights[0][None, :]
              layer_weights[f'{layer_no}']['bias'] = weights[1][None, :]
            elif 'lstm' in layer_name:
              layer_weights[f'{layer_no}']['name'] = 'lstm'
              layer_weights[f'{layer_no}']['input_weights'] = weights[0][None, :]
              layer_weights[f'{layer_no}']['recurrent_weights'] = weights[1][None, :]
              layer_weights[f'{layer_no}']['bias'] = weights[2][None, :]
            elif 'gru' in layer_name:
              layer_weights[f'{layer_no}']['name'] = 'gru'
              layer_weights[f'{layer_no}']['input_weights'] = weights[0][None, :]
              layer_weights[f'{layer_no}']['recurrent_weights'] = weights[1][None, :]
              layer_weights[f'{layer_no}']['bias'] = weights[2][None, :]
            else:
              raise ValueError("Unknown layer encountered while plotting weights.")            

          else:
            if 'dense' in layer_name:
              layer_weights[f'{layer_no}']['weights'] = np.concatenate((layer_weights[f'{layer_no}']['weights'], weights[0][None, :]))
              layer_weights[f'{layer_no}']['bias'] = np.concatenate((layer_weights[f'{layer_no}']['bias'], weights[1][None, :]))
            elif 'lstm' in layer_name:
              layer_weights[f'{layer_no}']['name'] = 'lstm'
              layer_weights[f'{layer_no}']['input_weights'] = np.concatenate((layer_weights[f'{layer_no}']['input_weights'], weights[0][None, :]))
              layer_weights[f'{layer_no}']['recurrent_weights'] = np.concatenate((layer_weights[f'{layer_no}']['recurrent_weights'], weights[1][None, :]))
              layer_weights[f'{layer_no}']['bias'] = np.concatenate((layer_weights[f'{layer_no}']['bias'], weights[2][None, :,:]))
            elif 'gru' in layer_name:
              layer_weights[f'{layer_no}']['name'] = 'gru'
              layer_weights[f'{layer_no}']['input_weights'] = np.concatenate((layer_weights[f'{layer_no}']['input_weights'], weights[0][None, :,:]))
              layer_weights[f'{layer_no}']['recurrent_weights'] = np.concatenate((layer_weights[f'{layer_no}']['recurrent_weights'], weights[1][None, :]))
              layer_weights[f'{layer_no}']['bias'] = np.concatenate((layer_weights[f'{layer_no}']['bias'], weights[2][None, :]))
            else:
              raise ValueError("Unknown layer encountered while plotting weights.")            
 
          
      # plot weights    
      for layer_no in layer_weights.keys():
        if layer_weights[layer_no]['name'] == 'dense':
          self.__plot_dense_weights(layer_weights[layer_no], f'{model.name}_dense_{layer_no}', vl = model_no)
        elif layer_weights[layer_no]['name'] == 'gru':
          self.__plot_gru_weights(layer_weights[layer_no],f'{model.name}_gru_{layer_no}', vl = model_no)
        elif layer_weights[layer_no]['name'] == 'lstm':
          self.__plot_lstm_weights(layer_weights[layer_no], f'{model.name}_lstm_{layer_no}', vl = model_no)

def __plot_dense_weights(self, values, title, vl):
    """
    Plots weights of dense layer
    
    Parameters:
      values: dict containing keys:
        weights: shape(num_iters, ...)
        bias: shape(num_iters, ...)
      title: titile of the figure and filename
      vl: index (model no) for plotting vertical lines
    """
    print(f"Plotting weights of dense layer {title}")
    
    weights = values['weights'] 
    bias = values['bias'] 

    num_iters = weights.shape[0]
        
    num_features = np.prod(weights.shape[1:]) + np.prod(bias.shape[1:])
    
    # setup number of rows
    num_rows = 15
    temp = num_features/num_rows
    if temp < 1:
      num_rows = num_features
    
    #set up number of cols
    num_columns = int(np.ceil(num_features/num_rows))
    
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15,15))
    plt.subplots_adjust(wspace=0.001, hspace=0)
    fig.suptitle(title+f'_weights')

    w = ['w', 'b']
    
    row_no = 0
    col_no = 0
    
    # control whether to plot weight or bias
    for j in range(2):
        
      if  j==0:
        data = weights.reshape(num_iters, -1)   
      if  j==1:  
        data = bias.reshape(num_iters, -1)

      
      for i in range(data.shape[-1]):
        # plot weights
        if num_columns == 1:
          axes[row_no].plot(data[:,i])#, label=f'{w[j]}{i}')
          axes[row_no].set_ylabel(f'{w[j]}{i}')
        else:  
          axes[row_no, col_no].plot(data[:,i])#, label=f'{w[j]}{i}')
          axes[row_no, col_no].set_ylabel(f'{w[j]}{i}')
        
        # plot vertical lines
        for l in self.vertical_lines[:, vl]:
          if num_columns == 1:
            axes[row_no].axvline(l, linestyle='--', color='black')
          else:
            axes[row_no, col_no].axvline(l)
        
        if num_columns ==1:
          axes[row_no].set_xticks([])
          #axes[row_no].legend()
        else:
          axes[row_no, col_no].set_xticks([])
          #axes[row_no, col_no].legend()
        
        row_no +=1
        if row_no == num_rows:
          if num_columns ==1:
            axes[row_no-1].set_xlabel(f'iteration')
            axes[row_no-1].set_xticks(np.arange(0, num_iters, int(num_iters/10)))    
          else:  
            axes[row_no-1, col_no].set_xlabel(f'iteration')
            axes[row_no-1, col_no].set_xticks(np.arange(0, num_iters, int(num_iters/10)))    
          row_no =0
          col_no+=1
      
    if row_no != 0:
      if num_columns ==1:
        axes[row_no].set_xlabel(f'iteration')
        axes[row_no].set_xticks(np.arange(0, num_iters, int(num_iters/10)))
      else:
        axes[row_no, col_no].set_xlabel(f'iteration')
        axes[row_no, col_no].set_xticks(np.arange(0, num_iters, int(num_iters/10)))
        
        while row_no != num_rows:
          axes[row_no, col_no].remove()
          row_no +=1

    fig.tight_layout()
  
    plt.savefig(os.path.join(self.path_to_results, title+f'_weights.png'))
    plt.close()

def __plot_gru_weights(self, values, title, vl):
    """
    Plots GRU layer weights
    
    Parameters:
      values: dict containing keys:
        input_weights: shape(num_iters, ...)
        recurrent_weights: shape(num_iters, ...)
        bias: shape(num_iters, ...)
      title: titile of the figure and filename
      vl: index (model no) for plotting vertical lines

    """
    print(f"Plotting weights of GRU layer {title} . . .")
    
    # get data
    input_weights = values['input_weights'] 
    recurrent_weights= values['recurrent_weights']  
    bias = values['bias'] 

    div = int(input_weights.shape[-1]/3)
    num_iters = input_weights.shape[0]
        
    num_features = int((np.prod(input_weights.shape[1:]) + np.prod(recurrent_weights.shape[1:]) + np.prod(bias.shape[1:]))/3) #in each gate
    
    num_rows = 15
    temp = num_features/num_rows
    if temp < 1:
      num_rows = num_features
    
    num_columns = int(np.ceil(num_features/num_rows))
    
    #gates = ['update', 'reset', 'memory']
    gates = ['u', 'r', 'm']
    w = ['w', 'rw', 'b']
    
    #control data of which gate to plot
    for i in range(3):
      fig, axes = plt.subplots(num_rows, num_columns, figsize=(15,15))
      plt.subplots_adjust(wspace=0.001, hspace=0)
      fig.suptitle(title+f'_{gates[i]}_weights')
      row_no = 0
      col_no = 0
    
      # control whether to plot weight, recurrent weight or bias
      for j in range(3):
          
        if  i == 0 and j==0:
          data = input_weights[:,:, :div].reshape(num_iters, -1)   
        if  i == 0 and j==1:   
          data = recurrent_weights[:, :, :div].reshape(num_iters, -1)
        if  i == 0 and j==2:  
          data = bias[:,:,:div].reshape(num_iters, -1)
        if  i == 1 and j==0:
          data = input_weights[:,:, div:div*2].reshape(num_iters, -1)   
        if  i == 1 and j==1:  
          data = recurrent_weights[:, :, div:div*2].reshape(num_iters, -1)
        if  i == 1 and j==2:  
          data = bias[:,:,div:div*2].reshape(num_iters, -1)
        if  i == 2 and j==0:
          data = input_weights[:,:, div*2:].reshape(num_iters, -1)   
        if  i == 2 and j==1:  
          data = recurrent_weights[:, :, div*2:].reshape(num_iters, -1)
        if  i == 2 and j==2:  
          data = bias[:,:,div*2:].reshape(num_iters, -1)
        

        
        for k in range(data.shape[-1]):
          # plot weight
          axes[row_no, col_no].plot(data[:,k])#, label=f'{gates[i]}_{w[j]}{k}')
          axes[row_no, col_no].set_ylabel(f'{gates[i]}_{w[j]}{k}')
          
          # plot vertical lines
          for l in self.vertical_lines[:, vl]:
            axes[row_no, col_no].axvline(l, linestyle='--', color='black')
          axes[row_no, col_no].set_xticks([])
          #axes[row_no, col_no].legend()
          row_no +=1
          if row_no == num_rows:
            axes[row_no-1, col_no].set_xlabel(f'iteration')
            axes[row_no-1, col_no].set_xticks(np.arange(0, num_iters, int(num_iters/10)))    
            row_no =0
            col_no+=1
        
      if row_no != 0:
        axes[row_no-1, col_no].set_xlabel(f'iteration')
        axes[row_no-1, col_no].set_xticks(np.arange(0, num_iters, int(num_iters/10)))

        while row_no != num_rows:
          axes[row_no, col_no].remove()
          row_no +=1
  
      fig.tight_layout()
    
      plt.savefig(os.path.join(self.path_to_results, title+f'_{gates[i]}_weights.png'))
      plt.close()

def __plot_lstm_weights(self, values, title, vl):
    """
    Plots LSTM layer weights
    
    Parameters:
      values: dict containing keys:
        input_weights: shape(num_iters, ...)
        recurrent_weights: shape(num_iters, ...)
        bias: shape(num_iters, ...)
      title: titile of the figure and filename
      vl: index (model no) for plotting vertical lines
    """
    print(f"Plotting weights of LSTM layer {title}")

    # get data
    input_weights = values['input_weights'] 
    recurrent_weights= values['recurrent_weights']  
    bias = values['bias']

    div = int(input_weights.shape[-1]/4)
    num_iters = input_weights.shape[0]
        
    num_features = int((np.prod(input_weights.shape[1:]) + np.prod(recurrent_weights.shape[1:]) + np.prod(bias.shape[1:]))/4)#in each gate
    
    num_rows = 10
    temp = num_features/num_rows
    if temp < 1:
      num_rows = num_features
    
    num_columns = int(np.ceil(num_features/num_rows))
    
    #gates = ['forget', 'input', 'output', 'cell']
    gates =['f', 'i', 'o', 'c']
    w = ['w', 'rw', 'b']
    
    #control data of which gate to plot
    for i in range(4):
      fig, axes = plt.subplots(num_rows, num_columns, figsize=(15,15))
      plt.subplots_adjust(wspace=0.001, hspace=0)
      fig.suptitle(title+f'_{gates[i]}_weights')
      row_no = 0
      col_no = 0
    
      # control whether to plot weight, recurrent weight or bias
      for j in range(3):
          
        if  i == 0 and j==0:
          data = input_weights[:,:, :div].reshape(num_iters, -1)   
        if  i == 0 and j==1:   
          data = recurrent_weights[:, :, :div].reshape(num_iters, -1)
        if  i == 0 and j==2:  
          data = bias[:,:,:div].reshape(num_iters, -1)
        if  i == 1 and j==0:
          data = input_weights[:,:, div:div*2].reshape(num_iters, -1)   
        if  i == 1 and j==1:  
          data = recurrent_weights[:, :, div:div*2].reshape(num_iters, -1)
        if  i == 1 and j==2:  
          data = bias[:,:,div:div*2].reshape(num_iters, -1)
        if  i == 2 and j==0:
          data = input_weights[:,:, div*2:div*3].reshape(num_iters, -1)   
        if  i == 2 and j==1:  
          data = recurrent_weights[:, :, div*2:div*3].reshape(num_iters, -1)
        if  i == 2 and j==2:  
          data = bias[:,:,div*2:div*3].reshape(num_iters, -1)
        if  i == 3 and j==0:
          data = input_weights[:,:, div*3:].reshape(num_iters, -1)   
        if  i == 3 and j==1:  
          data = recurrent_weights[:, :, div*3:].reshape(num_iters, -1)
        if  i == 3 and j==2:  
          data = bias[:,:,div*3:].reshape(num_iters, -1)
        

        
        for k in range(data.shape[-1]):
          # plot data
          axes[row_no, col_no].plot(data[:,k])#, label=f'{gates[i]}_{w[j]}{k}')
          axes[row_no, col_no].set_ylabel(f'{gates[i]}_{w[j]}{k}')
          
          # plot vertical lines
          for l in self.vertical_lines[:, vl]:
            axes[row_no, col_no].axvline(l, linestyle='--', color='black')
          
          axes[row_no, col_no].set_xticks([])
          axes[row_no, col_no].legend()
          
          row_no +=1
          if row_no == num_rows:
            axes[row_no-1, col_no].set_xlabel(f'iteration')
            axes[row_no-1, col_no].set_xticks(np.arange(0, num_iters, int(num_iters/10)))    
            row_no =0
            col_no+=1
        
      if row_no != 0:
        axes[row_no, col_no].set_xlabel(f'iteration')
        axes[row_no, col_no].set_xticks(np.arange(0, num_iters, int(num_iters/10)))
        while row_no != num_rows:
          axes[row_no, col_no].remove()
          row_no +=1

      fig.tight_layout()
    
      plt.savefig(os.path.join(self.path_to_results, title+f'_{gates[i]}_weights.png'))
      plt.close()



class Inference:
    
    def __init__(self, path_to_log, X, **kwargs):
        """
        Initializes the object

        Parameters:
          path_to_log: path to the log file containing paramters for the model
          X: original data of shape (num_samples, num_timesteps, num_features)
          Z: synthetic data of shape (num_samples, num_timesteps, num_features)
        """
        self.path_to_log = path_to_log
        self.X = torch.cuda.FloatTensor(X).cuda()
        self.setup_models()
        self.Z = torch.cuda.FloatTensor(np.random.normal(0, 1, (X.shape[0], self.latent_dim))).cuda()
      
    def setup_models(self):
        """
        Sets up models for TTS-GAN
        """
        with open(self.path_to_log, 'r') as f:
            log_data = f.read()

        # extract parameters from log file
        model_type = find_value_from_log('model_type', log_data)[1:-1]
        seq_len = int(find_value_from_log('seq_len', log_data))
        patch_size = int(find_value_from_log('patch_size', log_data))
        channels = int(find_value_from_log('channels', log_data))
        latent_dim = int(find_value_from_log('latent_dim', log_data))
        gf_dim = int(find_value_from_log('gf_dim', log_data))
        df_dim = int(find_value_from_log('df_dim', log_data))
        g_depth = int(find_value_from_log('g_depth', log_data))
        d_depth = int(find_value_from_log('d_depth', log_data))
        g_heads = int(find_value_from_log('g_heads', log_data))
        d_heads = int(find_value_from_log('d_heads', log_data))
        forward_dropout = float(find_value_from_log('forward_dropout', log_data))
        dropout = float(find_value_from_log('dropout', log_data))
        factor = int(find_value_from_log('factor', log_data))
        attn = find_value_from_log('attn', log_data)[1:-1]
        g_s_layers = find_value_from_log('g_s_layers', log_data)
        d_s_layers = find_value_from_log('d_s_layers', log_data)
        
        print(g_s_layers)
        print(d_s_layers)
        
        #TODO
        if model_type == 'transformer' or model_type=='informer':
            g_depth = g_depth
            d_depth = d_depth
        elif model_type=='informerstack':
            g_depth = [int(s_l) for s_l in g_s_layers.replace(' ','').replace("'", "").split(',')]
            d_depth = [int(s_l) for s_l in d_s_layers.replace(' ','').replace("'", "").split(',')]
            print(g_depth)
            print(d_depth)
        else:
            raise ValueError(f"Unknown type of model type encountered: {model_type}") 

        # set up model
        print(g_depth)
        print(d_depth)
        self.gen_net = Generator(seq_len=seq_len,
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
        self.gen_net.eval()

        self.dis_net = Discriminator(
                      in_channels=channels,
                      patch_size=patch_size,
                      emb_size=df_dim, 
                      seq_length = seq_len,
                      depth=d_depth, 
                      n_classes=1,
                      num_heads=d_heads,
                      
                      model_type=model_type,
                      
                      drop_p=dropout, 
                      forward_drop_p=forward_dropout, 
                      
                      factor= factor,
                      attn=attn,       
                      ).cuda()
        self.dis_net.eval()
        
        self.latent_dim = latent_dim
        
        # set up loss
        self.loss = find_value_from_log('loss', log_data)[1:-1]
        if self.loss != 'lsgan':
          raise ValueError("unknown loss encountered.")
        
    def load_weights(self, path_to_checkpoint):
      """
      Loads weights from the checkpoint

      Parameters:
          path_to_checkpoint 
      """
      checkpoint_dict = torch.load(path_to_checkpoint)
      _ = self.gen_net.load_state_dict(checkpoint_dict['gen_state_dict'])
      print("Generator: ", _)
      
      _ = self.dis_net.load_state_dict(checkpoint_dict['dis_state_dict'])
      print("Discrimiator: ", _)
           
    def infer(self):
      """
        Generates prediction and different kind of losses.

        Returns:
          X_hat: generated data of shape (num_samples, num_timesteps, num_features)
          d_loss: how good is the discriminator model in identifying real and fake data. (MSE(1, Y_real) + MSE(0, Y_fake))
          g_loss: how good the generator model is able to fool discriminator (MSE(1, Y_fake))       
      """
      X_hat = self.gen_net(self.Z)
      
      real_validity = self.dis_net(self.X)
      fake_validity = self.dis_net(X_hat)
      
      
      if self.loss == 'lsgan':
        # discriminator loss
        real_label = torch.full((real_validity.shape[0],real_validity.shape[1]), 1., dtype=torch.float, device=self.X.get_device())
        fake_label = torch.full((real_validity.shape[0],real_validity.shape[1]), 0., dtype=torch.float, device=self.X.get_device())
        d_real_loss = nn.MSELoss()(real_validity, real_label)
        d_fake_loss = nn.MSELoss()(fake_validity, fake_label)
        d_loss = d_real_loss + d_fake_loss
      
      
       # generator loss
        real_label = torch.full((fake_validity.shape[0],fake_validity.shape[1]), 1., dtype=torch.float, device=self.X.get_device())
        g_loss = nn.MSELoss()(fake_validity, real_label)
                        

      return X_hat.detach().cpu().numpy(), d_loss.detach().cpu().item(), g_loss.detach().cpu().item()
    

class TSNE_PCA:
    def __init__(self, X, sample_size, number_of_iters):
        """
        Initializes an object that can be used for calculation of tSNE and PCA results

        Parameters:
          X: original data (numpy array of shape(num_samples, num_timesteps, num_features))
          sample_size: number of samples to be used for the calculation 
          number_of_iters: number of iterations for tSNE
        """
        # reshape data
        self.X = np.transpose(X[:,:,0,:], (0,2,1))
        
        self.number_of_samples = self.X.shape[0]
        self.sample_size = sample_size
        self.number_of_iters = number_of_iters
        
    def get_results(self, X_hat, find_closest_samples):
        """
        Applies tSNE and PCA and returns results
        
        Parameters:
          X_hat: synthetic data of shape(num_samples, num_timesteps, num_features)
          find_closest_samples: whether to find closest samples or not

        Returns:
          tsne_results, pca_results: numpy array of shape (2*num_samples, 2). 
                  The first 'sample_size' indexes contain results for original data and remaining indexes for synthetic data 
          
          if find_closest_samples=True,
            tsne_ori_index, tsne_syn_index, pca_ori_index, pca_syn_index: list containing indexes for the closest samples
        
        Note:
          - tSNE and PCA only works with 2D data. Hence, the input (X/X_hat) is converted to 2D data by taking mean along features(axis=2)
          - Currently, only 2 closest samples are searched and returned.
        """
        num_closest_samples = 2
        
        # generate random samples of X and X_hat
        idx = np.random.permutation(self.number_of_samples)[:self.sample_size]
        X = self.X[idx]
        X_hat = X_hat[idx]

        # calculate mean (along features)
        X_mean = np.mean(X, 2)
        X_hat_mean = np.mean(X_hat, 2)

        # TSNE
        data = np.concatenate((X_mean, X_hat_mean), axis = 0)
        tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = self.number_of_iters)
        tsne_results = tsne.fit_transform(data)
        
        # PCA
        pca = PCA(n_components = 2)
        pca.fit(X_mean)
        pca_x_results = pca.transform(X_mean)
        pca_x_hat_results = pca.transform(X_hat_mean)
        pca_results = np.concatenate((pca_x_results, pca_x_hat_results), axis=0)
        
        if find_closest_samples:
           # find indexes for closest samples using tsne results
           tsne_ori_index = []
           tsne_syn_index = []
           temp_tsne_ori_index, temp_tsne_syn_index = self.__find_indexes_of_closest_samples(tsne_results[:self.sample_size, :], tsne_results[self.sample_size:, :], num_closest_samples)
           for i in range(num_closest_samples):
              tsne_ori_index.append(idx[temp_tsne_ori_index[i]])
              tsne_syn_index.append(idx[temp_tsne_syn_index[i]])
          
           # find indexes for closest samples using pca results
           pca_ori_index = []
           pca_syn_index = []
           temp_pca_ori_index, temp_pca_syn_index = self.__find_indexes_of_closest_samples(pca_x_results, pca_x_hat_results, num_closest_samples)
           for i in range(num_closest_samples):
              pca_ori_index.append(idx[temp_pca_ori_index[i]])
              pca_syn_index.append(idx[temp_pca_syn_index[i]])

           return tsne_results, pca_results, tsne_ori_index, tsne_syn_index, pca_ori_index, pca_syn_index
        
        else:
          return tsne_results, pca_results

    def __find_indexes_of_closest_samples(self,X, Y, number_of_samples):
      """
      Finds and returns indexes of closest samples by calculating pairwise euclidean distance between X and Y

      Parameters:
        X: numpy array of shape(num_obs, num_features)
        Y: numpy array of shape(num_obs, num_features)
        number_of_samples: number of closest samples to look for

      Returns:
        index_for_X: list containing indexes in X of closest samples
        index_for_Y: list containing indexes in Y of closest samples

      """
      index_in_X, index_in_Y = [], []

      # find pairwise distance
      distance = pairwise_distances(X,Y) # (num_obs_X, num_obs_Y)

      # sort the distances and get the closest samples
      samples_distances = np.sort(distance, axis=None)[:number_of_samples] 
      sds, counts = np.unique(samples_distances, return_counts=True)

      # find indexes of closest samples
      for sd, count in zip(sds, counts):
        
        # get ALL indexes where distance is equal to the current distance
        curr_index_in_X, curr_index_in_Y = np.where(distance == sd)
        
        if len(curr_index_in_X) != len(curr_index_in_Y):
           raise ValueError("The length of indexes in X is not equal to the length of indexes in Y.")
        
        if count !=1:
           print(f"Warning: Multiple indexes (count:{count}) found for the distance ({sd})")

        index_in_X.extend(curr_index_in_X[:count])
        index_in_Y.extend(curr_index_in_Y[:count])

      if len(index_in_X) != len(index_in_Y) != number_of_samples:
         raise ValueError("The number of indexes found in X and in Y does not equal to the number of samples required. ")

      return index_in_X, index_in_Y


def generate_data_for_training_visualisation(path_to_run, **kwargs):
  """
  Generates and saves data that can be used for the visualization of the training process.

  Parameters:
    path_to_run: path to the results folder

  Keywords:
    sample_size: to be used for creating PCA and tSNE plots (default: 1000)
    number_of_iters: max number of iters for the optimization in tSNE (default: 300)
    file_name: name of the file where data will be saved (default='training_visualization_data')

  Notes:
    - If the file already exists in the directory, it will not generate new data
    
  """ 
  file_name = kwargs.get('file_name', 'training_visualization_data')

  # set up path to the data
  path_to_processed_file = os.path.join(path_to_run, file_name)
  if os.path.exists(path_to_processed_file):
    print(f"File '{path_to_processed_file}' already exists. Skipping . . . ")
    return

  sample_size = kwargs.get('sample_size', 1000)
  number_of_iters = kwargs.get('number_of_iters', 300)
  
  print(f"Generating training visualization data . . . ")

  # load dataset
  path_to_data = os.path.join(path_to_run, 'Data', 'complete_dataset.npy')
  X = np.load(path_to_data)
  
  #set up path to log file
  path_to_log_folder = os.path.join(path_to_run, 'Log')
  log_file = [f for f in os.listdir(path_to_log_folder) if '_train.log' in f]
  path_to_log_file = os.path.join(path_to_log_folder, log_file[0])

  # get checkpoints
  path_to_checkpoints = os.path.join(path_to_run, 'Model')
  checkpoints = os.listdir(path_to_checkpoints)
  checkpoints_num = [int(cp[cp.rindex('_')+1:]) for cp in checkpoints]
  checkpoints_num = sort_cps(checkpoints_num)
  
  # set up tSNE and PCA
  tsne_pca = TSNE_PCA(X, sample_size, number_of_iters)
  tsne_results_list =[]
  pca_results_list = []
    
  # set up for generating synthetic data and loss values
  inference = Inference(path_to_log_file, X)
  discriminator_loss_list = []
  generator_loss_list = []
  
  temp_list = []
  for i, cp_no in enumerate(checkpoints_num):
    checkpoint = f'checkpoint_{cp_no}'
    print(f"Processing Checkpoint No: {cp_no}")

    # load weights
    path_to_checkpoint = os.path.join(path_to_checkpoints, checkpoint)
    inference.load_weights(path_to_checkpoint)
    
    # generate data and losses
    curr_X_hat, curr_discriminator_loss, curr_generator_loss = inference.infer()
    
    # reshape data
    curr_X_hat = np.transpose(curr_X_hat[:,:,0,:], (0,2,1))
        
    discriminator_loss_list.append(curr_discriminator_loss)
    generator_loss_list.append(curr_generator_loss)
    temp_list.append(curr_X_hat)
    
    if i == len(checkpoints_num) -1:
        curr_tsne_results, curr_pca_results, tsne_ori_index, tsne_syn_index, pca_ori_index, pca_syn_index =tsne_pca.get_results(curr_X_hat, True)

        # find data for closest samples
        tsne_closest_syn_sample_data = []
        tsne_closest_ori_sample_data = []
        pca_closest_syn_sample_data = []
        pca_closest_ori_sample_data = []

        for i in range(len(temp_list)):
          curr = temp_list[i]

          # using tsne results
          for j in range(len(tsne_syn_index)):
              if i == 0:
                  tsne_closest_syn_sample_data.append([])
                  temp_X = np.transpose(X[:,:,0,:], (0,2,1))
                  tsne_closest_ori_sample_data.append(temp_X[tsne_ori_index[j]])

              tsne_closest_syn_sample_data[j].append(curr[tsne_syn_index[j]])
          
          # using pca results
          for k in range(len(pca_syn_index)):
              if i == 0:
                  pca_closest_syn_sample_data.append([])
                  temp_X = np.transpose(X[:,:,0,:], (0,2,1))
                  pca_closest_ori_sample_data.append(temp_X[pca_ori_index[k]])
              
              pca_closest_syn_sample_data[k].append(curr[pca_syn_index[k]])

    else:
      curr_tsne_results, curr_pca_results =tsne_pca.get_results(curr_X_hat, False)
    
    tsne_results_list.append(curr_tsne_results) 
    pca_results_list.append(curr_pca_results) 
  
  xlabels = checkpoints_num
  # save dataset
  with open(path_to_processed_file, 'wb') as file:
      pickle.dump([generator_loss_list, discriminator_loss_list,xlabels,
                    tsne_results_list, pca_results_list, 
                    tsne_closest_ori_sample_data, tsne_closest_syn_sample_data, pca_closest_ori_sample_data, pca_closest_syn_sample_data], file)
  
  
def visualise_training_process(path_to_data, path_to_results, **kwargs):
  """
  Visualises training process by:
    1. creating an animation containing different losses, PCA and TSNE results
    2. creating an animation containing closest samples using PCA and TSNE results in the last cp.
  
  Parameters:
    path_to_data: path to the file where data for visulization is stored
    path_to_results: path to the folder where results will be saved
  
  Keywords:
    fps: frame per second (default=2)
    file_name: suffix to add to file names (default='')
    normalization_values: a dict containing normalization values. (default: no denormalization)
        Keys required: type, min and max or type, mean, std
         
    feature_names: list containing features names
  """
  fps = kwargs.get('fps', 2)
  normalization_values = kwargs.get('normalization_values', {})
  feature_names = kwargs.get('feature_names', [])
  
  # load data
  with open(path_to_data, 'rb') as file:
      [generator_loss_list, discriminator_loss_list, xlabels,
                    tsne_results_list, pca_results_list,
                    tsne_closest_ori_sample_data, tsne_closest_syn_sample_data, pca_closest_ori_sample_data, pca_closest_syn_sample_data] = pickle.load(file)
  
  # plot training loss, pca and tsne
  create_training_animation_for_loss_pca_tsne(generator_loss= generator_loss_list, 
                          discriminator_loss=discriminator_loss_list,
                          tsne_results=tsne_results_list, 
                          pca_results=pca_results_list, 
                          fps=fps,
                          xlabels=xlabels,
                          path_to_results=path_to_results)        
  
  # plot tsne closest samples
  for i in range(len(tsne_closest_syn_sample_data)):
    create_training_animation_for_closet_sample(syn_data=np.array(tsne_closest_syn_sample_data[i]),
                                                ori_data = np.array(tsne_closest_ori_sample_data[0]),
                                                fps=fps,
                                                path_to_results=path_to_results,
                                                file_name=f'tsne_{i}',
                                                normalization_values=normalization_values,
                                                feature_names=feature_names,
                                                epochs=xlabels)
  
  # plot pca closest samples
  for i in range(len(pca_closest_syn_sample_data)):
    create_training_animation_for_closet_sample(syn_data=np.array(pca_closest_syn_sample_data[i]),
                                                ori_data = np.array(pca_closest_ori_sample_data[0]),
                                                fps=fps,
                                                path_to_results=path_to_results,
                                                file_name=f'pca_{i}',
                                                normalization_values=normalization_values,
                                                feature_names=feature_names,
                                                epochs=xlabels)

def visualise_training_process_for_run(path_to_run):
    """
    Visualise training process for the given run

    Parameters:
      path_to_run: path to the results folder
    """

    path_to_animation_results = os.path.join(path_to_run, 'animation')
    if os.path.exists(path_to_animation_results):
      files = os.listdir(path_to_animation_results)
      cs_pca_counter, cs_tsne_counter, loss_counter = 0,0,0 
      for file in files:
        if 'closest_sample' in file:
            if 'pca' in file:
                cs_pca_counter +=1
            if 'tsne' in file:
                cs_tsne_counter +=1
        if 'loss' in file:
            loss_counter +=1

      if cs_tsne_counter == 2 and cs_pca_counter == 2 and loss_counter == 1:
        print(f"All visualization created. Skipping . . ")
        return
      
    
    print(f"Creating Visualization . . .")
    if not os.path.exists(path_to_animation_results):
      os.mkdir(path_to_animation_results)

    # get path to processed data
    #path_to_processed_data = pipeline[exp_name]['data']['path_to_processed_data']
    path_to_processed_data = path_to_run

    # load normalization values
    normalization_values = load_from_json(os.path.join(path_to_processed_data, 'Data', 'norm_values'))
    if normalization_values:
      if normalization_values['type'] == 'MinMax':
          normalization_values['type'] = 'min_max'
            
      if normalization_values['type'] == 'StandardScaler':
          normalization_values['type'] = 'standard_scaler'
      
    # get data name and features list
    path_to_log_folder = os.path.join(path_to_run, 'Log')
    log_file = [f for f in os.listdir(path_to_log_folder) if '_train.log' in f]
    path_to_log_file = os.path.join(path_to_log_folder, log_file[0])

    with open(path_to_log_file, 'r') as f:
        log_data = f.read()
    data_name = find_value_from_log('dataset', log_data)[1:-1]
    feature_names = [] 
    if data_name == 'stock_data':
      feature_names = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    if data_name == 'Running':
      feature_names = []
    if data_name  == 'particle_data':
      feature_names = ['FZ', 'FY', 'RotatingSpeed', 'angle_tangentiel', 'TC4', 'TC13', 'TC1', 'PM2.5']
    if data_name == 'sine' or data_name == 'cosine':
      feature_names = [] 
    
    path_to_data = os.path.join(path_to_run, 'training_visualization_data')

    
    visualise_training_process(path_to_data, path_to_animation_results, normalization_values=normalization_values, feature_names=feature_names)

if __name__ == "__main__":

  import sys

  path_to_results = 'logs'
  run_name_list = ['Running_with_transformer_model_2025_04_07_21_47_10',
  'Running_with_informer_model_2025_04_07_21_51_04',
  'Sine_with_transformer_model_2025_04_07_22_15_04',
  'Sine_with_informer_model_2025_04_07_22_28_24',
  'Stock_data_with_transformer_model_2025_04_07_22_35_53',
  #'particle_data_with_transformer_model_2025_04_07_22_57_50', #OOM error
  'Stock_data_with_informer_model_2025_04_07_22_58_14',
  'particle_data_with_informer_model_2025_04_07_23_31_34',
  'Running_with_informerstack_model_2025_04_08_00_05_36',
  'Sine_with_informerstack_model_2025_04_08_01_01_20',
  'Stock_data_with_informerstack_model_2025_04_08_01_51_07',
  'particle_data_with_informerstack_model_2025_04_08_02_42_46']
  
  number_of_runs = len(run_name_list)
  
  # generate data for visualisation
  for run_no in range(number_of_runs):
    run_name = run_name_list[run_no]
    path_to_run = os.path.join(path_to_results, run_name)
    
    print(f"Starting Run No: {run_no+1}/{number_of_runs} Run Name: {run_name} . . . ")
  
    start = time.time()
    sys.stdout=open(os.path.join(path_to_run, 'logs_cps.txt'), 'w')
    
    generate_data_for_training_visualisation(path_to_run)
    
    end = time.time()
    print(f"Time taken to visualise training process: {(end-start)/60} minutes")

    sys.stdout.close()
    sys.stdout = sys.__stdout__
  
  # create animation 
  for run_no in range(number_of_runs):
    run_name = run_name_list[run_no]
    path_to_run = os.path.join(path_to_results, run_name)
  
    print(f"Creating Visualization for Run: {run_name}")
    visualise_training_process_for_run(path_to_run)


  print("Processing Completed")


