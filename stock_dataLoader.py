"""
Sets up particle data for TTS-GAN

"""
import os
import pickle
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def apply_sliding_window(data, wl, ws):
        """
        Creates samples using window_length(wl) and window_stride(ws)
        
        Parameters:
            data: numpy array of shape (num_timestamps, num_features)
            wl: window_length
            ws: window stride
        
        Returns:
            windowed data: numpy array of shape (num_samples, wl, num_features)
        """
        number_of_snippets = ((data.shape[0] - wl) / ws) + 1

        output_shape = (int(number_of_snippets), wl, data.shape[-1])
        output_strides = (ws*data.strides[0], data.strides[0], data.strides[-1])  # in bytes

        data = np.lib.stride_tricks.as_strided(data, shape=output_shape, strides=output_strides)

        return data


class stock_load_dataset(Dataset):
    def __init__(self, 
        path_to_data, 
        window_length,
        window_stride = 1, 
        flip_data = True,
        is_normalize = False,
        norm_type = 'StandardScaler',
        norm_level = 'sample' 
        ):
        """
        Normalization:
        norm_type: kind of normalization: MinMax or StandardScaler
        norm_level: sample: at sample level, full: use complete dataset for calculating norm values
        
        """
        
        self.path_to_data = path_to_data 
        self.window_length = window_length
        self.window_stride = window_stride 

        self.flip_data = flip_data
        
        self.is_normalize = is_normalize
        
        assert norm_type in ['StandardScaler', 'MinMax']
        assert norm_level in ['sample', 'full']
        
        self.norm_type = norm_type
        self.norm_level = norm_level
        self.norm_values = {}
        
        self.__generate_stock_data()
        self.__adjust_for_pipeline()
        
    
    def __generate_stock_data(self):
        """
        Generates stock data 
        """
        print(f'Loading stock data . . . ')
        
        # read data from dir
        data = np.loadtxt(self.path_to_data, dtype=np.float32, delimiter=",", skiprows=1)
        
        # flip data
        if self.flip_data:
            print("Flipping the dataset . . .")
            data = data[::-1]

         # create snippets
        data = apply_sliding_window(data,  self.window_length, self.window_stride)
        
        # shuffle data
        index = np.random.permutation(data.shape[0])
        data = data[index] 

        print(f'Final Shape of the data: {data.shape}')
        self.data = data
                        
    def _normalize(self, epoch):
        """ A helper method for the normalization method.
            Returns
                result: a normalized epoch
        """
        e = 1e-10
        result = (epoch - epoch.mean(axis=0)) / ((np.sqrt(epoch.var(axis=0)))+e)
        return result
    
    def _min_max_normalize(self, epoch):
            
        result = (epoch - min(epoch)) / (max(epoch) - min(epoch))
        return result
    
    def normalization(self, epochs):
        """ Normalizes each epoch e s.t mean(e) = 0 and var(e) = 1 or using min max
        Args:
            epochs - Numpy structure of epochs
        Returns:
            epochs_n - mne data structure of normalized epochs (mean=0, var=1)
        """
        for i in range(epochs.shape[0]):
            for j in range(epochs.shape[1]):
                if self.norm_type == 'StandardScaler':
                    epochs[i,j,0,:] = self._normalize(epochs[i,j,0,:])
                if self.norm_type == 'MinMax':
                    epochs[i,j,0,:] = self._min_max_normalize(epochs[i,j,0,:])
        return epochs
    
    def _full_StandardScaler(self, x):
        """
        Normalizes features using complete dataset using mean and std

        Args:
            data: numpy array of shape (number_of_samples, number_of_features, 1, window_length) 
        """
        e = 1e-10
        x_mean = np.mean(x, axis=(0,2,3)).reshape(1,-1,1,1)
        x_var = np.var(x, axis=(0,2,3)).reshape(1,-1,1,1)
        x_std = np.sqrt(x_var) +e
        
        result = (x -x_mean)/x_std
        
        # set up norm dict
        self.norm_values['type'] = 'StandardScaler'
        self.norm_values['mean'] = x_mean.flatten().tolist()
        self.norm_values['std'] = x_std.flatten().tolist()
        
        return result
        
        
    def _full_MinMax(self, x):
        """
        Normalizes features using complete dataset using min and max

        Args:
            x: numpy array of shape (number_of_samples, number_of_features, 1, window_length) 
        """
        x_min = np.min(x, axis=(0,2,3)).reshape(1,-1,1,1)
        x_max = np.max(x, axis=(0,2,3)).reshape(1,-1,1,1)
        result = (x - x_min)/(x_max-x_min)
        
         
        # set up norm dict
        self.norm_values['type'] = 'MinMax'
        self.norm_values['min'] = x_min.flatten().tolist()
        self.norm_values['max'] = x_max.flatten().tolist()
        
        return result
    
    def full_normalization(self, data):
        """
        Performs normalization using complete dataset

        Args:
            data: (number_of_samples, number_of_features, 1, window_length) 
        Returns:
            normalized data: (number_of_samples, number_of_features, 1, window_length) 
        """  
        if self.norm_type == 'StandardScaler':
            output = self._full_StandardScaler(data)
        if self.norm_type == 'MinMax':
            output = self._full_MinMax(data)
            
        return output

    def get_norm_values(self):
        """
        Returns a dict containing norm values only if norm_type is 'full'
        """
        return self.norm_values
    
    def __adjust_for_pipeline(self):
        """
        Adjusts dataset for the TTS-GAN pipeline
        """
        # reshape data shape from (Batch, length, channels) to (Batch, channels, 1, length)
        self.data = np.transpose(self.data, (0, 2, 1))        
        self.data = self.data.reshape(self.data.shape[0], self.data.shape[1], 1, self.data.shape[2])

        # normalizes as follows:
        # take timesteps of one channel of one sample and use mean and var 
        # do it for all channels and all samples 
        # (Pixel Standardization in images- sample wise)
        if self.is_normalize:
            print(f"Normalizing data . . . level: {self.norm_level} type: {self.norm_type}")
            if self.norm_level == 'sample':
                self.data = self.normalization(self.data)
            else:
                self.data = self.full_normalization(self.data)
                
        self.number_of_samples = self.data.shape[0]
        print(f'Data shape after adjusting is {self.data.shape}')
    
    def save(self, path):
        """
        Saves dataset
        """
        np.save(os.path.join(path, 'complete_dataset.npy'), self.data)
       
    def __len__(self):
        return self.number_of_samples
    
    def __getitem__(self, idx):
        return self.data[idx], 0 # (X, Y)


if __name__ == "__main__":
    path_to_data = "/home/ahmad/Documents/Masters - TUHH/Research Project/timeGAN/data/stock_data.csv"
    
    window_length=24 
    
    upsample_Y = True 
    
    is_normalize = True
    norm_type ='MinMax' #'MinMax'#'StandardScaler'
    norm_level = 'full'

    dataset = stock_load_dataset(path_to_data, window_length, 
                                 is_normalize=is_normalize, 
                                 norm_level=norm_level,
                                 norm_type=norm_type)
    
    '''
    x = np.arange(3*5*4)
    np.random.shuffle(x)
    x = x.reshape(-1,4)
    
    print("Ori x")
    print(x)
    
    print("reshaped")
    x_sh = x.reshape(3,5,4)    
    x_sh= np.transpose(x_sh, (0, 2, 1))        
    x_sh = x_sh.reshape(x_sh.shape[0], x_sh.shape[1], 1, x_sh.shape[2])
    print(x_sh)
    
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    print("Min: ", x_min)
    print("Max: ", x_max)
    print("Mean: ", x_mean)
    print("Std: ", x_std)
    
    print("Minmax normalized")
    x_mm = (x - x_min)/(x_max-x_min)
    print(x_mm)
    print("from")
    print(dataset._full_MinMax(x_sh))
    print(dataset.get_norm_values())
    
    print("Stand norm")
    x_ss = (x-x_mean)/x_std
    print(x_ss)
    print("from")
    print(dataset._full_StandardScaler(x_sh))
    print(dataset.get_norm_values())
    '''
    
    
    
    
    #print(dataset[0])
    print(len(dataset))
    
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, 6, figsize=(20,5))
    
    
    train_loader = DataLoader(dataset, batch_size=32,shuffle = True)
    
    for x, _ in train_loader:
        #print(x)
        print(x.shape)
        
        for i in range(x.shape[0]):
            ax[0].plot(x[i, 0, 0,:], linewidth=1)
            ax[1].plot(x[i, 1, 0,:], linewidth=1)
            ax[2].plot(x[i, 2, 0,:], linewidth=1)
            ax[3].plot(x[i, 3, 0,:], linewidth=1)
            ax[4].plot(x[i, 4, 0,:], linewidth=1)
            ax[5].plot(x[i, 5, 0,:], linewidth=1)
    
    plt.tight_layout()
    plt.savefig("logs/stock_pictures/full_MinMax.png")
    print("Processing Completed.")
    
    '''
    Find args from log file
    
     train_set = stock_load_dataset(
            args.data_path,
            args.seq_len,
            is_normalize = args.is_normalize,
            norm_type = args.norm_type,
            norm_level = args.norm_level 
        )
        
    path_to_run = ""
    path_to_data = os.path.join(path_to_run_data, 'Data')
    
    # save norm values NOTE:ADDED BY AHMAD
    path_to_norm_values = os.path.join(args.path_helper['data_path'],'norm_values')
    save_to_json(train_set.get_norm_values(), path_to_norm_values)
    
    # save dataset
    train_set.save(args.path_helper['data_path'])
    '''