import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import safe_mask
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from numpy.random import randint
from collections import deque

def get_storm_indices(data, stormtimes_df, 
                      include_storms, time_resolution='5T'):
    stormtimes_list = [
        pd.date_range(t['start_time'].round(time_resolution),
                      t['end_time'], freq=time_resolution)
        for _, t in stormtimes_df.loc[include_storms].iterrows()
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

        self._lag_feature_processors = [
            OutputLagFeatureProcessor(y, auto_order)
        ]
        self._lag_feature_processors.extend([
            InputLagFeatureProcessor(data, order, delay)
            for data, order, delay in zip(X.T, exog_order, exog_delay)
        ])

    def generate_lag_features(self):
        lag_feature_list = [
            p.generate_lag_features() for p in self._lag_feature_processors
        ]

        if self.auto_order == 0:
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
