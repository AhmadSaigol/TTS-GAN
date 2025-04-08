"""
Sets up particle data for TTS-GAN

"""
import os
import pickle
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader

class trigonometric_load_dataset(Dataset):
    def __init__(self, 
        number_of_samples, 
        window_length, 
        number_of_features,
        trig_fnt,
        is_normalize = False,
        norm_type = 'StandardScaler',
        norm_level = 'sample'
        ):
        """
        Normalization:
        norm_type: kind of normalization: MinMax or StandardScaler
        norm_level: sample: at sample level, full: use complete dataset for calculating norm values
        
        """
        self.number_of_samples = number_of_samples
        self.window_length = window_length 
        self.number_of_features = number_of_features
        self.trig_fnt = trig_fnt
        
        self.is_normalize = is_normalize
        
        assert norm_type in ['StandardScaler', 'MinMax']
        assert norm_level in ['sample', 'full']
        
        self.norm_type = norm_type
        self.norm_level = norm_level
        self.norm_values = {}
        
        self.__generate_trigonometric_data()
        self.__adjust_for_pipeline()
        
     
    def __generate_trigonometric_data(self):
        """
        Generates and returns trigonometric data of shape (number_of_samples, window_length, number_of_features)
        """
        print(f"Generating {self.trig_fnt} data . . .")
        
        wd_dict = {}
        
        # set timestamps 
        ts = np.arange(self.window_length).reshape(-1,1)
        #repeat curr_ts in each feature and in each sample
        #[[0], [1]] -> [[[0,0],[1,1]], [[0,0],[1,1]]] (no=2, seq=2, fs=2)
        ts = np.tile(ts, (self.number_of_samples, 1, self.number_of_features))

        # generate random frequencies and phases
        freqs = np.random.uniform(0, 0.1, size=(self.number_of_samples, 1, self.number_of_features))
        phases = np.random.uniform(0, 0.1, size=(self.number_of_samples, 1, self.number_of_features))
        
        wd_dict['frequencies'] = freqs.tolist()
        wd_dict['phases'] = phases.tolist()
        
        # find radians
        radians = freqs * ts + phases

        # apply trigonometric function
        if self.trig_fnt == 'sine':
            data = np.sin(radians)
        elif self.trig_fnt == 'cosine':
            data = np.cos(radians)

        data = data.astype(np.float32)
        
        print(f'Final Shape of the data: {data.shape}')
        
        self.data = data
        
        # return data, wd_dict
    
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

        #self.number_of_samples = self.data.shape[0]
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
    
    number_of_features = 5
    number_of_samples = 10000
    window_length=24
    trig_fnt = 'cosine'#'cosine'#'sine'
    
    is_normalize = True
    norm_type ='StandardScaler' #'MinMax'#'StandardScaler'
    norm_level = 'full' #sample
  

    dataset = trigonometric_load_dataset(number_of_samples, 
                                         window_length, 
                                         number_of_features, 
                                         trig_fnt, 
                                         is_normalize=is_normalize,
                                         norm_level=norm_level,
                                         norm_type=norm_type)
    
    #print(dataset[0])
    print(len(dataset))
    
    train_loader = DataLoader(dataset, batch_size=32,shuffle = True)
    
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, 5, figsize=(20,5))
    
    for x, _ in train_loader:
        #print(x)
        print(x.shape)
        for i in range(x.shape[0]):
            ax[0].plot(x[i, 0, 0,:], linewidth=1)
            ax[1].plot(x[i, 1, 0,:], linewidth=1)
            ax[2].plot(x[i, 2, 0,:], linewidth=1)
            ax[3].plot(x[i, 3, 0,:], linewidth=1)
            ax[4].plot(x[i, 4, 0,:], linewidth=1)
    
    plt.tight_layout()
    plt.savefig("logs/trig_pictures/full_StandardScaler_cosine.png")
    print("Processing Completed.")