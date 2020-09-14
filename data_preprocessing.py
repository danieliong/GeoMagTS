from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from GeoMagTS.utils import get_storm_indices, MetaLagFeatureProcessor, shift
import matplotlib.pyplot as plt
import warnings

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column_names=None):
        """Subset columns in pandas DataFrame 

        Parameters
        ----------
        column_names : list of str, optional
            A sequence of column names, by default None
        """
        self.column_names = column_names

    def fit(self, X, y=None):
        # Check if column_names are in X
        if not all(col in X.columns for col in self.column_names):
            raise ValueError("Not all names in column_names are in X.")
        return self

    def transform(self, X, y=None):
        if self.column_names is None:
            return X
        else:
            return X[self.column_names]

class TimeResolutionResampler(BaseEstimator, TransformerMixin):
    def __init__(self, time_resolution='5T', func=np.mean):
        """Changes time resolution of data

        Parameters
        ----------
        time_resolution : DateOffset, Timedelta, or str, optional 
            Frequency string, by default '5T'
        func : function, optional
            Function for aggregating data, by default np.mean
        """
        self.time_resolution = time_resolution
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.resample(self.time_resolution).apply(self.func)

class StormsProcessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 storm_times_df=None,
                 storms_to_use=None,
                 storms_to_delete=None,
                 start='1980',
                 end='2020',
                 target_column='sym_h',
                 time_resolution='5T',
                 vx_colname='vx_gse',
                 D=1500000,
                 min_threshold=None,
                 interpolate=True,
                 interpolate_method='time'):
        self.storm_times_df = storm_times_df
        self.storms_to_use = storms_to_use
        self.storms_to_delete = storms_to_delete
        self.start = start
        self.end = end
        self.target_column = target_column
        self.time_resolution = time_resolution
        self.min_threshold = min_threshold
        self.interpolate = interpolate
        self.interpolate_method = interpolate_method

    def fit(self, X, y=None):
        self.columns_ = X.columns
        self.times_ = X.index

        if self.storm_times_df is not None:
            # TODO: Error handling
            times_to_include = pd.date_range(start=self.start,
                                             end=self.end,
                                             freq=self.time_resolution,
                                             closed='left')
            self.storm_times_df = self.storm_times_df[
                (self.storm_times_df['start_time'] >= times_to_include[0]) &
                (self.storm_times_df['end_time'] <= times_to_include[-1])
            ]

            if self.storms_to_use is None and self.storms_to_delete is None:
                self.storms_to_use_ = self.storm_times_df.index
            elif self.storms_to_delete is not None:
                # TODO: Check storms_to_delete are in storm_times_df.index
                self.storms_deleted_ = self.storm_times_df.index.intersection(
                    self.storms_to_delete)
                self.storms_to_use_ = self.storm_times_df.index.drop(
                    self.storms_deleted_)
            else:
                self.storms_to_use_ = self.storm_times_df.index.intersection(
                    self.storms_to_use)

            self.storm_indices_ = get_storm_indices(
                X, self.storm_times_df, self.storms_to_use_, time_resolution=self.time_resolution)

        return self

    def _transform_one_storm(self, X, storm_idx):
        X_one_storm = X.iloc[self.storm_indices_[
            storm_idx]].assign(storm=storm_idx)

        if self.interpolate:
            X_one_storm = X_one_storm.interpolate(
                method=self.interpolate_method,
                axis=0, limit_direction='both')

        return X_one_storm

    def transform(self, X, y=None):
        if self.storm_times_df is not None:
            X_processed = pd.concat(
                [self._transform_one_storm(X, i)
                 for i in range(len(self.storm_indices_))]
            )
            X_processed.reset_index(inplace=True)
            X_processed.set_index(['storm', 'times'], inplace=True)

            return X_processed
        else:
            return X

    # TODO: Delete later
    def get_propagated_times(self, vx_colname='vx_gse', D=1500000):

        def _get_propagated_time(x):
            return x.times + pd.Timedelta(x[vx_colname])

        vx = self.data_[vx_colname]
        prop_times = pd.Series(D / np.abs(vx), name='prop_times')
        propagated_times = prop_times.reset_index().apply(
            lambda x: (x.times + pd.Timedelta(x['prop_times'], unit='sec')).floor(freq=self.time_resolution), axis=1)
        return propagated_times

    def get_column_names(self):
        return self.columns_

    def get_target_column(self):
        return self.target_column_

    # TODO: plot_storms function
    def plot_storms(self):
        pass

class StormSplitter(BaseEstimator, TransformerMixin):
    def __init__(self,
                 test_storms=None,
                 min_threshold=None,
                 test_size=1,
                 storm_level=0,
                 target_column='sym_h'):
        # TODO: Input validation
        self.test_storms = test_storms
        self.min_threshold = min_threshold
        self.test_size = test_size
        self.storm_level = storm_level
        self.target_column = target_column

    def fit(self, X, y=None):
        # TODO: Input validation
        if isinstance(self.target_column, int):
            y = X.iloc[:, self.target_column]
        elif isinstance(self.target_column, str):
            y = X[self.target_column]

        self.storm_labels_ = y.index.unique(level=self.storm_level)

        if self.test_storms is None:

            if self.min_threshold is None:
                self.test_storms_ = np.random.choice(
                    self.storm_labels_, self.test_size)
            else:
                min_y = y.groupby(level=self.storm_level).min()
                self.storms_thresholded_ = self.storm_labels_[
                    min_y < self.min_threshold]
                self.test_storms_ = np.random.choice(
                    self.storms_thresholded_, self.test_size)

            self.train_storms_ = self.storm_labels_.drop(
                self.test_storms_).to_numpy()
        else:
            if not np.in1d(self.test_storms, self.storm_labels_).all():
                raise ValueError("Not all of test_storms are in y.")

            self.train_storms_ = self.storm_labels_.drop(
                self.test_storms).to_numpy()

        return self

    def transform(self, X, y=None):
        check_is_fitted(self)

        if isinstance(self.target_column, int):
            y_ = X.iloc[:, self.target_column]
        elif isinstance(self.target_column, str):
            y_ = X[self.target_column]
        else:
            raise TypeError("target_column must be type int or str.")

        X_ = X.drop(self.target_column, axis=1)

        if self.test_storms is None:
            test_idx = pd.IndexSlice[self.test_storms_, :]
        else:
            test_idx = pd.IndexSlice[self.test_storms, :]
        train_idx = pd.IndexSlice[self.train_storms_, :]

        # Return dictionary?
        # data = {
        #     'X_train': X.iloc[train_idx],
        #     'y_train': y.iloc[train_idx],
        #     'X_test': X.iloc[test_idx],
        #     'y_test': y.iloc[test_idx]
        # }

        X_train, y_train = X_.loc[train_idx, :], y_.loc[train_idx]
        X_test, y_test = X_.loc[test_idx, :], y_.loc[test_idx]
        return X_train, y_train, X_test, y_test

    def get_train_test_storms(self):
        check_is_fitted(self)
        if self.test_storms is None:
            return self.train_storms_, self.test_storms_
        else:
            return self.train_storms, self.test_storms_

def prepare_geomag_data(data, storm_times_df, 
                        test_storms=None, 
                        min_threshold=None,
                        test_size=1,
                        time_resolution='5T', 
                        target_column='sym_h',
                        feature_columns=['bz','vx_gse','density'],
                        storms_to_delete=[15, 69, 124],
                        start='2000',
                        end='2030',
                        split_train_test=True):

    if test_storms is None and min_threshold is None:
        raise ValueError(
            "Either test_storms or min_threshold must be specified. Specify only one and try again."
            )
    elif test_storms is not None and min_threshold is not None:
        raise ValueError(
            "test_storms and min_threshold cannot both be specified. Specify only one and try again."
            )

    # Data processing pipeline for entire dataframe
    column_selector = DataFrameSelector([target_column]+feature_columns)
    time_res_resampler = TimeResolutionResampler(time_resolution)
    storms_processor = StormsProcessor(storm_times_df=storm_times_df,
                                       storms_to_delete=storms_to_delete,start=start,
                                       end=end)
    
    pipeline_transformers = [
        ("selector", column_selector),
        ("resampler", time_res_resampler),
        ("processor", storms_processor)
        ]
    
    if split_train_test:
        storm_splitter = StormSplitter(test_storms=test_storms,
                                       min_threshold=min_threshold,test_size=test_size)
        pipeline_transformers.append(
            ("splitter", storm_splitter))
    
    data_pipeline = Pipeline(pipeline_transformers)
    processed_data = data_pipeline.fit_transform(data)
    
    return processed_data
