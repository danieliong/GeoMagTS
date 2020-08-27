import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import safe_mask
from sklearn.model_selection import GroupShuffleSplit
from numpy.random import randint

def get_storm_indices(data, stormtimes_df, include_storms, time_resolution='5T'):
    stormtimes_list = [
        pd.date_range(t['start_time'].round(time_resolution), t['end_time'], freq=time_resolution)
        for _, t in stormtimes_df.iloc[include_storms].iterrows()
        ]
    
    try:
        return [np.where(data.reset_index(level=0)['times'].isin(times))[0] for times in stormtimes_list]
    except KeyError:
        return [np.where(data.reset_index(level=0)['index'].isin(times))[0] for times in stormtimes_list]

def _get_NA_mask(X, y=None):
    if y is None:
        mask = safe_mask(X, np.isnan(X).any(axis=1))
    else:
        data_ = np.concatenate([y.reshape(-1, 1), X], axis=1)
        mask = np.isnan(data_).any(axis=1)    
        # Check mask
        mask = safe_mask(X, mask)
        mask = safe_mask(y, mask)
        
    return mask

def create_narx_model(n_hidden, learning_rate,
                      activation_hidden='tanh',
                      activation_output='linear',
                      optimizer=keras.optimizers.RMSprop,
                      loss=keras.losses.MeanSquaredError(),
                      metrics=[keras.metrics.MeanSquaredError()]
                      ):

    model = Sequential(
        [Dense(n_hidden, activation=activation_hidden,
                    use_bias=False),
         Dense(1, activation=activation_output, use_bias=False)]
    )
    model.compile(optimizer=optimizer(learning_rate),
                    loss=loss, metrics=metrics)
    return model            

def get_min_y_storms(y, storm_labels):
    
    def _get_min_y_one_storm(y, storm_labels, i):
        idx = (storm_labels == i)
        return np.amin(y[idx], where=~np.isnan(y[idx]), initial=0)
    
    unique_storms = np.unique(storm_labels)
    min_y =  np.array([_get_min_y_one_storm(y, storm_labels, i) 
                       for i in unique_storms]
                      )
    return min_y
    

class trainTestStormSplit():
    def __init__(self, storm_labels, test_storms=None,
                 min_threshold=None, y=None, test_size=1):
        # TODO: Input validation
        self.storm_labels = storm_labels
        self.test_storms = test_storms
        self.min_threshold = min_threshold
        self.test_size = test_size
        self.unique_storm_labels = np.unique(storm_labels)
        
        if self.test_storms is None and self.min_threshold is not None:
            # Pick only storms with y < min_threshold to be in test set 
            if y is None:
                raise ValueError(
                    "y must be specified if min_threshold is given."
                    )
                
            min_y = get_min_y_storms(y, self.storm_labels)
            storms_thres = self.unique_storm_labels[min_y < min_threshold]
            self.test_storms = np.random.choice(storms_thres, self.test_size)
        elif self.test_storms is None and self.min_threshold is None:
            # Choose random storm for testing
             self.test_storms = np.random.choice(self.unique_storm_labels, self.test_size)   
        # If test_storms is given, just use it
        self._split_idx_by_test_storms()
        
    def _split_idx_by_test_storms(self):
        if self.test_storms is None:
            raise ValueError("self.test_storms is None.")
        test_mask = np.in1d(self.storm_labels, self.test_storms)
        self.train_idx = np.where(~test_mask)[0]
        self.test_idx = np.where(test_mask)[0]
        
    def split_data(self, X, y):
        # TODO: Input validation
        X_train, y_train = X[self.train_idx,:], y[self.train_idx]
        X_test, y_test = X[self.test_idx,:], y[self.test_idx]        
        return X_train, y_train, X_test, y_test
    
    def split_storm_labels(self):
        return self.storm_labels[self.train_idx], self.storm_labels[self.test_idx]

    def get_test_storm_times(self, times):
        # TODO: Input validation
        return times[self.test_idx]
    
    
        
        
        
    
    
            
    
