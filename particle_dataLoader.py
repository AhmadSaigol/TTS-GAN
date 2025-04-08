"""
Sets up particle data for TTS-GAN

"""
import os
import pickle
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader

def load_from_json(path_to_json):
    """
    Load a json file

    Parameters:
        path_to_json: path to json file
    
    Returns:
        dictionary with contents of the json
    """
    with open(path_to_json +".json", 'r') as f:
        dic = json.load(f)
    
    return dic

def get_train_test_files_from_txt(path_to_txt, which_folder_data):
    """
    Finds and return training and test files
    """
    if which_folder_data not in [ 'QTI', 'REF', 'CTI', 'ITI']:
        raise ValueError(f"Unknown name of the folder({which_folder_data}) encountered.")
    
    train_files_flag = False
    test_files_flag =False
    save_file_flag = False

    train_files_list =[]
    test_files_list = []
    with open(path_to_txt, 'r') as log:
        while(1):
            # read single line from file
            line= log.readline()
            if line:
                # determine whether files are of train data or test data
                if 'Files in Training Data:' in line:
                    train_files_flag = True
                    test_files_flag = False
                    continue

                if 'Files in Test Data:' in line:
                    train_files_flag = False
                    test_files_flag = True
                    continue
        
                
                if train_files_flag or test_files_flag:
                    
                    if '.h5' in line:
                        
                        # get file name
                        file_name = line.split('/')[-1].replace('\n', '')

                        # determine whether to store current file or not
                        if which_folder_data == 'REF':
                            if 'ref' in file_name or 'Ref' in file_name:
                                save_file_flag = True
                        elif which_folder_data in ['QTI', 'ITI', 'CTI']:
                            if which_folder_data in file_name:
                                save_file_flag = True
                        
                        # store current file
                        if save_file_flag:
                            if train_files_flag:
                                train_files_list.append(file_name)
                            if test_files_flag:
                                test_files_list.append(file_name)
                        
                        save_file_flag = False
            else:
                break #EOF

    return train_files_list, test_files_list

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

def upsample_array(data, window_length):
        """
        Up samples an array

        Parameters:
            data: numpy array of shape (num_samples, num_features)

        Returns:
            numpy array of shape (num_samples, window_length, num_features)

        Notes:
            - Up samples the array with the same logic as that of mode = 'constant' in '__upsample_table_data'
        """

        # find constant values for all features in upsampled data
        constant_values = data/window_length
    
        # up sample data
        data_upsampled = np.repeat(constant_values, window_length, axis=0).reshape(-1, window_length, data.shape[-1])

        return data_upsampled


class particle_load_dataset(Dataset):
    def __init__(self, 
        path_to_data, 
        which_folder_data, 
        fold_no, 
        window_length, 
        upsample_Y,
        is_normalize = False,
        norm_type = 'StandardScaler',
        norm_level = 'sample'):
        
        """
        Normalization:
        norm_type: kind of normalization: MinMax or StandardScaler
        norm_level: sample: at sample level, full: use complete dataset for calculating norm values
        
        """
        
        self.path_to_data = path_to_data 
        self.which_folder_data = which_folder_data  
        self.fold_no = fold_no  
        self.window_length = window_length 
        self.upsample_Y = upsample_Y
        
        self.is_normalize = is_normalize
        
        assert norm_type in ['StandardScaler', 'MinMax']
        assert norm_level in ['sample', 'full']
        
        self.norm_type = norm_type
        self.norm_level = norm_level
        self.norm_values = {}
        
        self.__generate_particle_data()
        self.__adjust_for_pipeline()
        
    
    def __generate_particle_data(self):
        """
        Generates particles data 
        """
        # load data
        if self.which_folder_data == 'complete':
            X, y = self.__load_processed_particle_data()
        else:
            X, y = self.__create_processed_particle_data()

        
        # denormalize data
        print("Denormalizing particle data . . .")
        x_norm_dict = load_from_json(os.path.join(self.path_to_data, 'input_normalization_values'))
        y_norm_dict = load_from_json(os.path.join(self.path_to_data, 'output_normalization_values'))

        x_min_values = np.array(x_norm_dict['min'])
        x_max_values = np.array(x_norm_dict['max'])
        y_min_values = np.array(y_norm_dict['min'])
        y_max_values = np.array(y_norm_dict['max'])
        
        X = X*(x_max_values-x_min_values) + x_min_values
        y = y*(y_max_values-y_min_values) + y_min_values
    
        # get active channels
        print("Loading channels from the directory . . .")
        path_to_channels = os.path.join(self.path_to_data, 'active_channels.txt')
        with open(path_to_channels, 'r') as f:
            raw_channels = f.read().splitlines()
        
        # filter data using thresholds
        #if self.threshold_samples:
        #    X, y = self.__remove_samples_using_threshold(X, y, raw_channels)
        #else:
        #    self.gs_dict['threshold_samples'] = False
        
        channels = []
        for channel in raw_channels:
            if '(I)' in channel:
                channels.append(channel.replace('(I)', ''))
            elif '(O)' in channel:
                channels.append(channel.replace('(O)', ''))
            else:
                raise ValueError(f'Unknown channel ({channel}) found in the active_channels.txt')

        
        # up sample data
        if self.upsample_Y:
            print("Upsampling Y . . . ")
            # y shape (samples, bins) -> (samples, timesteps, bins) 
            y = upsample_array(y, self.window_length)

        data = np.concatenate((X, y), axis=-1, dtype=np.float32) 
    
        print(f'Final Shape of the data: {data.shape}')


        # shuffle dataset
        index = np.random.permutation(data.shape[0])
        data = data[index] 
        self.data = data
                
    
    def __load_processed_particle_data(self):
        """
        Loads processed data and returns X, y (train)
        """
        print("Loading processsed data . . .")   
        path_to_processed_data = os.path.join(self.path_to_data, 'processed_data')
        
        with open(path_to_processed_data, 'rb') as file:
            [X_train, _, y_train, _] = pickle.load(file)
        
        X = X_train[self.fold_no]
        y = y_train[self.fold_no]
        return X, y
 
    def __create_processed_particle_data(self):
        """
        Creates processed particle data using the files in input and output folder for the given fold number and folder name
        
        Returns:
            particle_data_processed: 
            X : numpy array of shape(number_of_samples, window_length, number_of_features)
            Y: numpy array of shape(number_of_samples, number_of_features)
        """
        # get files in training data
        path_to_txt = os.path.join(self.path_to_data, f'fold_{self.fold_no}_braking_file_names.txt')

        train_files_list, _ = get_train_test_files_from_txt(path_to_txt, self.which_folder_data)
        print(f"Creating processed particle data . . . Folder: '{self.which_folder_data}' Files Found: {len(train_files_list)}")
        
        num_of_zero_output_snippets_removed = 0
        X, Y = [], []

        # load normalization values
        x_norm_dict = load_from_json(os.path.join(self.path_to_data, 'input_normalization_values'))
        x_min_values = np.array(x_norm_dict['min'])
        x_max_values = np.array(x_norm_dict['max'])
        y_norm_dict = load_from_json(os.path.join(self.path_to_data, 'output_normalization_values'))
        y_min_values = np.array(y_norm_dict['min'])
        y_max_values = np.array(y_norm_dict['max'])
         
        for file in train_files_list:
            # change .h5 to .pkl
            file = file.replace('.h5', '.pkl')

            # load input data
            path_to_input_file = os.path.join(self.path_to_data, 'files', 'input', file) 
            with open(path_to_input_file, 'rb') as f:
                df_X = pd.read_pickle(f)

            # load output data
            path_to_output_file = os.path.join(self.path_to_data, 'files', 'output', file)
            with open(path_to_output_file, 'rb') as f:
                df_Y = pd.read_pickle(f)
            
            print(f"Loaded file {file} data successfully. Shape: df_X:{df_X.shape} df_Y:{df_Y.shape}")

            # create snippets
            print("Creating snippets . . .")
            x_curr = apply_sliding_window(np.array(df_X), self.window_length, self.window_length)                
            y_curr = np.array(df_Y)
            
            # remove zero snippets
            non_zero_snippets_indexes = np.sum(y_curr, axis=1) != 0
            x_curr = x_curr[non_zero_snippets_indexes]
            y_curr = y_curr[non_zero_snippets_indexes]
            curr_num_of_zero_output_snippets_removed = len(non_zero_snippets_indexes) - sum(non_zero_snippets_indexes)
            num_of_zero_output_snippets_removed += curr_num_of_zero_output_snippets_removed
            print(f"Removed {curr_num_of_zero_output_snippets_removed} zero output snippets.")
        
            if len(x_curr) !=0:
                X.append(x_curr)
                Y.append(y_curr)
            else:
                print(f"No data from the file would be used as output data doesn't contain any data.")

        print("Total zero output snippets removed: ", num_of_zero_output_snippets_removed)
        
        # combine dataset
        X=np.vstack(X)
        Y=np.vstack(Y)

        print("Normalizing X,Y . . . ")
        X = (X - x_min_values)/(x_max_values-x_min_values)
        Y = (Y - y_min_values)/(y_max_values-y_min_values)
        
        # shuffle dataset
        indexes = np.random.permutation(X.shape[0])
        X =X[indexes]
        Y= Y[indexes]

        return X, Y

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
    
    def normalization(self, epochs):
        """ Normalizes each epoch e s.t mean(e) = 0 and var(e) = 1
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
    path_to_data = "/home/ahmad/Documents/Masters - TUHH/Research Project/timeGAN/data/PM2.5_bins_split_type_kfold_files_REFs_QTIs"
    which_folder_data = 'complete'
    fold_no = 1
    window_length=300 
    upsample_Y = True 
    
    is_normalize = True
    norm_type ='StandardScaler' #'MinMax'#'StandardScaler'
    norm_level = 'full' #'sample', 'full'


    dataset = particle_load_dataset(path_to_data, 
                                    which_folder_data, 
                                    fold_no, 
                                    window_length, 
                                    upsample_Y, 
                                    is_normalize=is_normalize,
                                    norm_level=norm_level,
                                    norm_type=norm_type)
    
    #print(dataset[0])
    print(len(dataset))
    
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(20,10))
    
    PARTICLE_DATA_CHANNELS = ['FZ', 'FY', 'RotatingSpeed', 'angle_tangentiel', 'TC4', 'TC13', 'TC1', 'PM2.5']
    
    train_loader = DataLoader(dataset, batch_size=32,shuffle = True)
    
    for x, _ in train_loader:
        #print(x)
        print(x.shape)
        
        for i in range(x.shape[0]):
            ax[0, 0].plot(x[i, 0, 0,:], linewidth=0.5)
            ax[0, 1].plot(x[i, 1, 0,:], linewidth=0.5)
            ax[0, 2].plot(x[i, 2, 0,:], linewidth=0.5)
            ax[0, 3].plot(x[i, 3, 0,:], linewidth=0.5)
            ax[1, 0].plot(x[i, 4, 0,:], linewidth=0.5)
            ax[1, 1].plot(x[i, 5, 0,:], linewidth=0.5)
            ax[1, 2].plot(x[i, 6, 0,:], linewidth=0.5)
            ax[1, 3].plot(x[i, 7, 0,:], linewidth=0.5)
    
    ax[0, 0].set_title(PARTICLE_DATA_CHANNELS[0])
    ax[0, 1].set_title(PARTICLE_DATA_CHANNELS[1])
    ax[0, 2].set_title(PARTICLE_DATA_CHANNELS[2])
    ax[0, 3].set_title(PARTICLE_DATA_CHANNELS[3])
    ax[1, 0].set_title(PARTICLE_DATA_CHANNELS[4])
    ax[1, 1].set_title(PARTICLE_DATA_CHANNELS[5])
    ax[1, 2].set_title(PARTICLE_DATA_CHANNELS[6])
    ax[1, 3].set_title(PARTICLE_DATA_CHANNELS[7])
    
    plt.tight_layout()
    plt.savefig("logs/particle_data_figures/full_StandardScaler_complete.png")
    
    
    '''
    Find args from log file
    
    train_set = particle_load_dataset(
        args.data_path,
        args.which_folder_data,
        args.fold_no,
        args.seq_len,
        args.upsample_Y,
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
    
    print("Processing Completed.")