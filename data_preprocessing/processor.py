import warnings
import numpy as np
import pandas as pd
import copy

from GeoMagTS.utils import _read_data, is_pd_freqstr, read_storm_times, check_storm_data
from GeoMagTS.data_preprocessing.sklearn import MovingAverageSmoother, StormSplitter

class GeoMagDataProcessor():
    def __init__(self, 
                 data_file=None,
                 **kwargs,
                 ):
        self.data_file = data_file 
        
        self.read_data_kwargs = kwargs 
        self._data = _read_data(data_file, **self.read_data_kwargs)
        
        self.time_resolution = self._check_time_resolution(infer_freq=True)
        
        self.columns_selected = False
        self.time_resolution_resampled = False 
        self.interpolated = False 
        self.storms_processed = False  
            
    def _check_time_resolution(self, time_resolution=None, 
                              infer_freq=False):
        
        if time_resolution is None:
            if infer_freq:
                time_resolution = self._data.index.inferred_freq
                if time_resolution is None:
                    warnings.warn(
                        "Frequency could not be inferred from self._data.")
            else:
                raise ValueError("time_resolution is None and infer_freq is False.")
        elif not isinstance(time_resolution, str):
            raise TypeError("time_resolution must be a string.")
        elif not is_pd_freqstr(time_resolution):
            raise TypeError("time_resolution is not a valid Pandas frequency string.")
        
        return time_resolution
    
    def select_columns(self, 
                       target_column=None,
                       feature_columns=None,
                       inplace=True,
                       data=None):
        
        # If inplace is True, data is ignored if specified. 
        if inplace or data is None:
            data = self._data
        
        if target_column is None and feature_columns is None:
            return data  
        
        if not isinstance(target_column, str):
            raise TypeError("target_column must be a string.")
        if not isinstance(feature_columns, tuple):
            raise TypeError("feature_columns must be a tuple.")
        
        if isinstance(feature_columns, tuple):
            columns = list((target_column,)+feature_columns)
        elif isinstance(self.features, list):
            columns = [target_column]+feature_columns
        
        if not all(col in data.columns for col in columns):
            raise ValueError("Not all specified columns are in data columns.")
        
        if inplace:
            self._data = data[columns]
            self.columns_selected = True
            return None 
        else:
            return data[columns]
        
    
    def resample_time_resolution(self, 
                                 time_resolution,
                                 resample_func=np.mean,
                                 inplace=True, data=None, **kwargs):
        
        if inplace or data is None:
            data = self._data
            
        time_resolution = self._check_time_resolution(time_resolution)
        _data = data.resample(time_resolution).apply(resample_func)
        
        if inplace:
            self._data = _data
            self.time_resolution = time_resolution
            self.resample_func_ = resample_func
            self.time_resolution_resampled = True
            return None 
        else:
            return _data
    
    def interpolate(self, method='time', limit_direction='both', 
                    inplace=True, data=None, **kwargs):
        
        if inplace or data is None:
            data = self._data
        
        def interp_one_storm(storm):
            return storm.reset_index(
                level=0, drop=True).interpolate(
                method=method, axis=0, 
                limit_direction=limit_direction, **kwargs)
            
        if self.storms_processed:
            _data = data.groupby(level=0).apply(interp_one_storm)
        else:
            _data = data.interpolate(
                method=method, axis=0, limit_direction=limit_direction, **kwargs)
        
        if inplace: 
            self._data = _data
            self.interpolate_method = method
            self.interpolate_kwargs = kwargs
            self.interpolated = True
            return None
        else:
            return _data
        
    def process_storms(self,
                       storm_times_file=None,
                       start='1990',
                       end='2030',
                       storms_to_delete=None, 
                       storms_to_use=None,
                       inplace=True,
                       data=None,
                       **kwargs):
        
        if inplace or data is None:
            data = self._data
        
        if storm_times_file is None:
            warnings.warn(
                "storm_times_file is not specified. Nothing will be done.")
            if inplace:
                return None
            else:
                return data
        
        storm_times_df_ = read_storm_times(
            storm_times_file=storm_times_file,
            start=start, end=end,
            storms_to_delete=storms_to_delete,
            storms_to_use=storms_to_use,
            **kwargs)
        
        # storm_indices_ = get_storm_indices(
        #     self._data, storm_times_df_, self.time_resolution)
        
        def subset_one_storm(storm_row):
            # NOTE: Can actually use between_time in pd 
            time_format = '%Y-%m-%d %H:%M:%S'
            start_str = storm_row['start_time'].strftime(time_format)
            end_str = storm_row['end_time'].strftime(time_format)
            return data.loc[start_str:end_str].assign(
                storms=storm_row.name)
        
        _data = pd.concat(
            [subset_one_storm(storm_row) 
             for _, storm_row in storm_times_df_.iterrows()]
        )
        _data.reset_index(inplace=True)
        _data.set_index(['storms','times'], inplace=True)
        
        if inplace:
            self._data = _data
            self.storm_times_df_ = storm_times_df_
            self.storms_processed = True 
            return None 
        else:
            return _data           
    
    def smooth(self, method='simple', window=30, inplace=True, 
               data=None, **kwargs):
        if inplace or data is None:
            data = self._data
        
        smoother = MovingAverageSmoother(
            method=method,
            window=window,
            time_resolution=self.time_resolution,
            **kwargs)
        
        if inplace:
            self._data = smoother.fit_transform(data)
            return None
        else:
            self_copy = copy.deepcopy(self)
            self_copy._data = smoother.fit_transform(data)
            return self_copy
    
    def train_test_split(self, test_storms=None,
                         min_threshold=None,
                         test_size=None, data=None, 
                         return_dict=True, **kwargs):
        if data is None:
            if not self.storms_processed:
                raise ValueError(
                    "Storms must be processed via process_storms before splitting.")
            else:
                data = self._data
        else:
            check_storm_data(data)

        storm_splitter = StormSplitter(
            test_storms=test_storms,
            min_threshold=min_threshold,
            test_size=test_size, return_dict=return_dict, 
            **kwargs)
        split_data = storm_splitter.fit_transform(data)
        return split_data
