import itertools
import warnings
from collections import deque
from functools import lru_cache
from os import path

import keras
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from numpy.random import randint
from pandas.api.types import is_datetime64_dtype, is_list_like
from pandas.tseries.frequencies import to_offset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.utils import safe_mask
from sklearn.utils.validation import check_is_fitted


# @lru_cache(maxsize=2048)
def _read_data(data_file, **kwargs):
    if data_file is None:
            raise ValueError("Data file must be specified.")
    elif not isinstance(data_file, str):
        raise TypeError("Data file must be a string.")
    elif (data_file.endswith('pkl') or
            data_file.endswith('pickle')):
        data = pd.read_pickle(data_file, **kwargs)
    elif data_file.endswith('csv'):
        data = pd.read_csv(data_file, **kwargs)
    elif data_file.endswith('txt'):
        data = pd.read_table(data_file, **kwargs)
    else:
        raise ValueError(
            "Data file must have extension pkl, pickle, csv, or txt.")
    
    if not is_datetime64_dtype(data.index):
        raise TypeError("Index must be of type datetime64.")
    
    return data 

@lru_cache(maxsize=2048)
def read_storm_times(storm_times_file, start='1990', end='2030', storms_to_delete=None, storms_to_use=None, **kwargs):
        
        if storm_times_file is None:
            raise ValueError("storm_times_file must be specified.")
        
        # Read in storm times depending on what type of file it is.
        if not isinstance(storm_times_file, str):
            raise TypeError("Storm times file must be a string.")
        elif (storm_times_file.endswith('pkl') or
              storm_times_file.endswith('pickle')):
            storm_times = pd.read_pickle(storm_times_file, **kwargs)
        elif storm_times_file.endswith('csv'):
            storm_times = pd.read_csv(storm_times_file, **kwargs)
        elif storm_times_file.endswith('txt'):
            storm_times = pd.read_table(storm_times_file, **kwargs)
        else:
            raise ValueError(
                "Storm times file must have extension pkl, pickle, csv, or txt.")
        
        storm_times = check_process_storm_times(storm_times)
        
        # Filter storms_to_use and drop storms_to_delete
        if storms_to_use is not None:
            storms_to_use = storm_times.index.intersection(storms_to_use)
            storm_times = storm_times.loc[storms_to_use]
        if storms_to_delete is not None:
            storms_to_delete = storm_times.index.intersection(storms_to_delete)
            storm_times = storm_times.drop(storms_to_delete)
        
        # Subset by start and end
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)
        storm_times = storm_times[
            (storm_times['start_time'] >= start) &
            (storm_times['end_time'] <= end) 
        ]
        
        
        return storm_times

def is_pd_freqstr(freqstr): 
    freqstr_list = pd.tseries.frequencies._offset_to_period_map.values()
    return (freqstr[-1] in freqstr_list)    

def check_process_storm_times(storm_times_df):
    # Check if start_time and end_time are columns of dtype datetime64
    if ('start_time' not in storm_times_df.columns or
        'end_time' not in storm_times_df.columns):
        raise ValueError(
            'Data in storm_times_file must have columns start_time and end_time indicating start and end time of storms.')
    elif (not is_datetime64_dtype(storm_times_df['start_time']) or
            not is_datetime64_dtype(storm_times_df['end_time'])):
        raise TypeError(
            "start_time and end_time must have dtype datetime64.")
    
    if len(storm_times_df.columns) > 2:
        storm_times_df = storm_times_df[['start_time','end_time']]
    
    return storm_times_df

def check_storm_data(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pd.DataFrame.")
    elif not isinstance(data.index, pd.MultiIndex):
        raise TypeError("Data's index must be a pd.MultiIndex.")
    elif not data.index.nlevels != 2:
        raise ValueError("Data's index must have two levels.")
    else:
        levels = data.index.levels
        if levels[0].dtype != 'int':
            raise TypeError("First level of data's index must have dtype int.")
        elif not is_datetime64_dtype(levels[1]):
            raise TypeError("Second level of data's index must have dtype datetime64.")
    
    return None 

def find_storms(data, times, storm_times_df=None, date_only=True):
    if type(data).__name__ == 'GeoMagDataProcessor':
        if data.storms_processed:
            storm_times_df = data.storm_times_df_
        elif storm_times_df is None:
            raise ValueError(
                "storm_times_df must be specified if data is a GeoMagDataProcessor object with storms_processed=False.")
    else:
        if storm_times_df is None:
            raise ValueError(
                "storm_times_df must be specified if data is not a GeoMagDataProcessor object.")
        storm_times_df = check_process_storm_times(storm_times_df)            
    
    if date_only:
        storm_times_df = storm_times_df.applymap(lambda x: x.date())
    
    storms = []
    def get_storms_at_time(t):
        if isinstance(t, str):
            t = pd.Timestamp(t)
        if date_only:
            t = t.date()
            
        for i, start_end in storm_times_df.iterrows():
            if start_end['start_time'] <= t and start_end['end_time'] >= t:
                storms.append(i)
    
    if is_list_like(times):
        if isinstance(times, tuple):
            times = list(times)
        
        for t in times:
            get_storms_at_time(t)
    else:
        get_storms_at_time(times)          
        
    return storms

def _check_file_name(file_name, valid_ext=None):
    import re
    name, ext = file_name.split('.')
    
    if valid_ext is not None and ext not in valid_ext:
        valid_ext_str = ', '.join(valid_ext)
        raise ValueError("File extension must be "+valid_ext_str+".")         
    
    if path.exists(file_name):
        name_search = re.search('\d+$', name)
        if name_search is not None:
            name = name_search.re.split(name)[0]
            num = str(int(name_search.group())+1)
            name = name + num
        else:
            name = name + '2'
        
        file_name = name + '.' + ext
    
    return file_name

def get_storm_indices(data, stormtimes_df, 
                      include_storms=None, time_resolution='5T'):
    if include_storms is not None:
        stormtimes_df = stormtimes_df.loc[include_storms]
        
    stormtimes_list = [
        pd.date_range(t['start_time'].round(time_resolution),
                      t['end_time'], freq=time_resolution)
        for _, t in stormtimes_df.iterrows()
    ]

    try:
        return [np.where(data.reset_index(level=0)['times'].isin(times))[0] for times in stormtimes_list]
    except KeyError:
        return [np.where(data.reset_index(level=0)['index'].isin(times))[0] for times in stormtimes_list]

def _get_NA_mask(X, y=None, mask_y=False):    
    if y is None:
        mask_y = False
    
    if mask_y:
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_ = y.to_numpy()
            data_ = np.concatenate([y_.reshape(-1,1), X], axis=1)
        else:
            data_ = np.concatenate([y.reshape(-1, 1), X], axis=1)
    else:
        data_ = X
    
    mask = ~np.isnan(data_).any(axis=1)
    return mask

def _pd_fit(transformer, X, y=None, **transformer_fit_params):
    if transformer is not None:
        if isinstance(X, pd.Series):
            X_np = X.to_numpy().reshape(-1, 1)
            return transformer.fit(X_np, **transformer_fit_params)
        elif isinstance(X, pd.DataFrame):
            return transformer.fit(X, **transformer_fit_params)
        else:
            raise TypeError(
                "_pd_fit is meant to only take in pandas objects as input.")
    else:
        return None

def _pd_transform(transformer, X, y=None):
    if transformer is not None:
        if isinstance(transformer, Pipeline):
            # Check each step of Pipeline
            for step in transformer.steps:
                check_is_fitted(step[1])
        else:
            check_is_fitted(transformer)

        if isinstance(X, pd.Series):
            # Need to reshape
            X_transf = transformer.transform(
                X.to_numpy().reshape(-1, 1)).flatten()
            X_transf = pd.Series(X_transf, index=X.index)
        elif isinstance(X, pd.DataFrame):
            X_transf = transformer.transform(X)
            X_transf = pd.DataFrame(X_transf, index=X.index)
        else:
            raise TypeError(
                "_pd_transform is meant to only take in pandas objects as input.")

        return X_transf
    else:
        # Do nothing
        return X

def mse_masked(y, ypred, squared=True, round=False, copy=True, **kwargs):
    y_ = y
    ypred_ = ypred
    if copy:
        y_ = y_.copy()
        ypred_ = ypred.copy()
    
    y_ = y_.reindex(ypred.index)
    nan_mask = ~np.isnan(y_)
    mse = mean_squared_error(y_[nan_mask], ypred_[nan_mask],
                            squared=squared, **kwargs)
    if round:
        mse = np.around(mse, decimals=3)

    return mse

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

def get_min_y_storms(y, storm_labels=None):

    def _get_min_y_one_storm(y, storm_labels, i):
        idx = (storm_labels == i)
        return np.amin(y[idx], where=~np.isnan(y[idx]), initial=0)

    unique_storms = np.unique(storm_labels)
    min_y = np.array([_get_min_y_one_storm(y, storm_labels, i)
                      for i in unique_storms]
                     )
    return min_y

# THE FUNCTIONS BELOW WERE TAKEN FROM FIRETS (https://github.com/jxx123/fireTS) AND MODIFIED SLIGHTLY

def shift(darray, k, axis=0):
    """
    Utility function to shift a numpy array

    Inputs
    ------
    darray: a numpy array
        the array to be shifted.
    k: integer
        number of shift
    axis: non-negative integer
        axis to perform shift operation

    Outputs
    -------
    shifted numpy array, fill the unknown values with nan
    """
    if k == 0:
        return darray
    elif k < 0:
        shift_array = np.roll(darray, k, axis=axis).astype(float)
        shift_array[k:] = np.nan
        return shift_array
    else:
        shift_array = np.roll(darray, k, axis=axis).astype(float)
        shift_array[:k] = np.nan
        return shift_array


class OutputLagFeatureProcessor:
    def __init__(self, data, order):
        self._feature_queue = deque([shift(data, l) for l in range(order)])

    def generate_lag_features(self):
        return np.array(self._feature_queue).T

    def update(self, data_new):
        # TODO: this is not memory efficient, need to do this in a
        # better way in the future
        self._feature_queue.appendleft(data_new)
        self._feature_queue.pop()
        return np.array(self._feature_queue).T


class InputLagFeatureProcessor:
    def __init__(self, data, order, delay):
        self._data = data
        self._lags = np.array(range(delay, delay + order))

    def generate_lag_features(self):
        features = [shift(self._data, l) for l in self._lags]
        return np.array(features).T

    def update(self):
        self._lags = self._lags - 1
        return self.generate_lag_features()


class MetaLagFeatureProcessor(object):
    def __init__(self, X, y, auto_order, exog_order, exog_delay):

        self.auto_order = auto_order
        self.exog_order = np.array(exog_order)
        if auto_order == 0 and exog_order.count(0) == len(exog_order):
            raise ValueError("auto_order and exog_order are all 0.")

        self.y_exists_ = False if y is None else True
        self._lag_feature_processors = []

        if self.y_exists_:
            self._lag_feature_processors.append(
                OutputLagFeatureProcessor(y, auto_order))

        self._lag_feature_processors.extend([
            InputLagFeatureProcessor(data, order, delay)
            for data, order, delay in zip(X.T, exog_order, exog_delay)
        ])

    def generate_lag_features(self):
        lag_feature_list = [
           p.generate_lag_features() for p in self._lag_feature_processors
        ]

        if self.auto_order == 0 and self.y_exists_:
            lag_features = np.concatenate(lag_feature_list[1:],
                                          axis=1)
        elif np.all(self.exog_order == 0):
            lag_features = np.array(lag_feature_list[0])
        else:
            lag_features = np.concatenate(lag_feature_list,
                                          axis=1)
        return lag_features

    def update(self, data_new):
        lag_feature_list = [
            p.update(data_new) if i == 0 else p.update()
            for i, p in enumerate(self._lag_feature_processors)
        ]
        lag_features = np.concatenate(lag_feature_list, axis=1)
        return lag_features
