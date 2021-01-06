import warnings
import numpy as np
import pandas as pd
import copy

from GeoMagTS.utils import _read_data, is_pd_freqstr, read_storm_times, check_storm_data
from GeoMagTS.data_preprocessing.sklearn import MovingAverageSmoother, StormSplitter

class GeoMagDataProcessor():
    def __init__(self, 
                 data_file=None,
                 data=None,
                 **kwargs,
                 ):
        """Data processing object 

        Parameters
        ----------
        data_file : str, optional
            Name of file that contains data, by default None
        data : pandas.DataFrame, optional
            Data, by default None

        Raises
        ------
        ValueError
            Both data_file and data are specified.
        ValueError
            Both data_file and data are not specified.
        """        
        self.data_file = data_file 

        if self.data_file is not None and data is not None:
            raise ValueError(
                "Only one of data_file and data can be specified.")
        elif self.data_file is None and data is None:
            raise ValueError(
                "One of data_file and data must be specified."
            )
            
        if self.data_file is not None:
            self.read_data_kwargs = kwargs 
            self.data = _read_data(data_file, **self.read_data_kwargs)
        else:
            self.data = data
        # elif self.data is not None:
        
        self.time_resolution = self._check_time_resolution(infer_freq=True)
        
        # Attributes to keep track of processing steps
        self.columns_selected = False
        self.time_resolution_resampled = False 
        self.interpolated = False 
        self.storms_processed = False  
    
    # TODO
    def _check_data(self, data):
        pass
    
    def _check_time_resolution(self, time_resolution=None, 
                              infer_freq=False):
        
        if time_resolution is None:
            if infer_freq:
                time_resolution = self.data.index.inferred_freq
                if time_resolution is None:
                    warnings.warn(
                        "Frequency could not be inferred from self.data.")
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
                       data=None,
                       return_data=False):
        """Subset columns in self.data or data

        Parameters
        ----------
        target_column : str, optional
            Name of target column in self.data or data, by default None
        feature_columns : tuple, optional
            Name of feature_columns to subset, by default None
        inplace : bool, optional
            Change self.data in place and set self.columns_selected to True, by default True
        data : pandas.DataFrame, optional
            Data to be processed, by default self.data
        return_data : bool, optional
            If True, return data instead of GeomagDataProcessor object, by default False

        Returns
        -------
        GeoMagDataProcessor or pandas.DataFrame
            See return_data

        Raises
        ------
        TypeError
            target_column is not a string.
        TypeError
            feature_columns is not a tuple.
        ValueError
            Not all specified columns are in the data.
        """        
        # If inplace is True, data is ignored if specified. 
        if inplace or data is None:
            data = self.data
        
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
        
        _data = data[columns]
        
        if inplace:
            self.data = _data
            self.columns_selected = True
            return None 
        else:
            if return_data:
                return _data
            else:
                new_processor = copy.deepcopy(self)
                new_processor.data = _data
                return new_processor
        
    def resample_time_resolution(self, 
                                 time_resolution,
                                 resample_func=np.mean,
                                 inplace=True, data=None, 
                                 return_data=False, **kwargs):
        """Change time resolution in data.

        Parameters
        ----------
        time_resolution : str, datetime.timedelta, or pandas.DateOffset
            Desired time resolution
        resample_func : function, optional
            Function used to resample data, by default np.mean
        inplace : bool, optional
            Change self.data in place and set self.time_resolution_resampled to True, by default True
        data : pandas.DataFrame, optional
            Data to be processed, by default self.data
        return_data : bool, optional
            If True, return data instead of GeomagDataProcessor object, by default False

        Returns
        -------
        GeoMagDataProcessor or pandas.DataFrame
            See return_data
        """
        
        if inplace or data is None:
            data = self.data
            
        time_resolution = self._check_time_resolution(time_resolution)
        _data = data.resample(time_resolution).apply(resample_func)
        
        if inplace:
            self.data = _data
            self.time_resolution = time_resolution
            self.resample_func_ = resample_func
            self.time_resolution_resampled = True
            return None 
        else:
            if return_data:
                return _data
            else:
                new_processor = copy.deepcopy(self)
                new_processor.data = _data
                return new_processor
    
    def interpolate(self, method='time', limit_direction='both', 
                    inplace=True, data=None, return_data=False, 
                    **kwargs):
        """Interpolate missing values in data

        Parameters
        ----------
        method : str, optional
            Interpolation technique to use. see pandas.DataFrame.interpolate, by default 'time'
        limit_direction : {{'forward','backward','both'}}, optional
            See pandas.DataFrame.interpolate, by default 'both'
        inplace : bool, optional
            Change self.data in place and set self.interpolated to True, by default True
        data : pandas.DataFrame, optional
            Data to be processed, by default self.data
        return_data : bool, optional
            If True, return data instead of GeomagDataProcessor object, by default False

        Returns
        -------
        GeoMagDataProcessor or pandas.DataFrame
            See return_data
        """
        
        if inplace or data is None:
            data = self.data
        
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
            self.data = _data
            self.interpolate_method = method
            self.interpolate_kwargs = kwargs
            self.interpolated = True
            return None
        else:
            if return_data:
                return _data
            else:
                new_processor = copy.deepcopy(self)
                new_processor.data = _data
                return new_processor
        
    def process_storms(self,
                       storm_times_file=None,
                       start='1990',
                       end='2030',
                       storms_to_delete=None, 
                       storms_to_use=None,
                       inplace=True,
                       data=None,
                       return_data=False,
                       **kwargs):
        """Process and subset data by storms.

        Parameters
        ----------
        storm_times_file : str, optional
            Name of file containing storm times. Must have columns named
            'start_time' and 'end_time', by default None
        start : datetime-like or str, optional
            Start time, by default '1990'
        end : datetime-like or str, optional
            End time , by default '2030'
        storms_to_delete : array-like, optional
            Storms to delete, by default None
        storms_to_use : array-like, optional
            Storms to use, by default None
        inplace : bool, optional
            Change self.data in place and set self.storms_processed to True, by default True
        data : pandas.DataFrame, optional
            Data to be processed, by default self.data
        return_data : bool, optional
            If True, return data instead of GeomagDataProcessor object, by default False

        Returns
        -------
        GeoMagDataProcessor or pandas.DataFrame
            If inplace, self.data becomes pandas.DataFrame with pd.MultiIndex
            where the first level is storms and second level is time.
        """
        if inplace or data is None:
            data = self.data
        
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
            self.data = _data
            self.storm_times_df_ = storm_times_df_
            self.storms_processed = True 
            return None 
        else:
            if return_data:
                return _data
            else:
                new_processor = copy.deepcopy(self)
                new_processor.data = _data
                return new_processor
    
    def smooth(self, method='simple', window=30, inplace=True, 
               data=None, return_data=False, **kwargs):
        if inplace or data is None:
            data = self.data
        
        smoother = MovingAverageSmoother(
            method=method,
            window=window,
            time_resolution=self.time_resolution,
            **kwargs)
        
        _data = smoother.fit_transform(data)
        
        if inplace:
            self.data = _data
            return None
        else:
            if return_data:
                return _data
            else:
                new_processor = copy.deepcopy(self)
                new_processor.data = _data
                return new_processor
    
    def train_test_split(self, test_storms=None,
                         min_threshold=None,
                         test_size=None, data=None,
                         return_dict=True, seed=None, **kwargs):
        """Split self.data into training and testing set. 
        
        There are several options for how to split the data. You can either
        specify the specific storms you want to use for testing in test_storms
        or specify the test size and split the data randomly. If min_threshold
        is specified, the random test storms will be chosen among storms with a
        min. value < min_threshold.   

        Parameters
        ----------
        test_storms : list or tuple of int, optional
            Specific storms to be used for testing, by default None
        min_threshold : float, optional
            Minimum value threshold, by default None
        test_size : int or float, optional
            If < 1, it will be considered a percentage. If > 1, it will be
            considered as the number of test storms, by default None
        data : pandas.DataFrame, optional
            Data to split, by default self.data
        return_dict : bool, optional
        Return two dictionaries for training and test with keys 'X' and 'y', by
        default True. If False, return tuples with entries X_train, y_train,
        X_test, y_test
        seed : int, optional
            Random seed for generating test indices, by default None 

        Returns
        -------
        dict or tuple
            See return_dict

        Raises
        ------
        ValueError
            self.data is not processed using process_storms first. 
        """        
        if data is None:
            if not self.storms_processed:
                raise ValueError(
                    "Storms must be processed via process_storms before splitting.")
            else:
                data = self.data
        else:
            check_storm_data(data)            

        storm_splitter = StormSplitter(
            test_storms=test_storms,
            min_threshold=min_threshold,
            test_size=test_size, return_dict=return_dict, 
            seed=seed, **kwargs)
        split_data = storm_splitter.fit_transform(data)
        return split_data
