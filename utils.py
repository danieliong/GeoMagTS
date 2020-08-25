import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np

def get_storm_indices(data, stormtimes_df, include_storms, time_resolution='5T'):
    stormtimes_list = [
        pd.date_range(t['start_time'].round(time_resolution), t['end_time'], freq=time_resolution)
        for _, t in stormtimes_df.iloc[include_storms].iterrows()
        ]
    
    try:
        return [np.where(data.reset_index(level=0)['times'].isin(times))[0] for times in stormtimes_list]
    except KeyError:
        return [np.where(data.reset_index(level=0)['index'].isin(times))[0] for times in stormtimes_list]

def create_narx_model(n_hidden, learning_rate):
        model = Sequential()
        model.add(Dense(n_hidden, activation='tanh', use_bias=False))
        model.add(Dense(1, activation='linear', use_bias=False))
        model.compile(optimizer=keras.optimizers.RMSprop(learning_rate),             loss=keras.losses.MeanSquaredError(),
                      metrics=[keras.metrics.MeanSquaredError()])
        return model
    
