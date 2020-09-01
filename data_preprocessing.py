from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.utils import safe_mask, check_scalar
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array, check_consistent_length

import numpy as np
import pandas as pd
from GeoMagTS.utils import get_storm_indices, MetaLagFeatureProcessor, shift
import matplotlib.pyplot as plt

# IDEA: Might be better to change to handling pd dataframes instead of np arrays


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column_names=None):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.column_names is None:
            return X
        else:
            return X[self.column_names]


class timeResolutionResampler(BaseEstimator, TransformerMixin):
    def __init__(self, time_resolution='5T', func=np.mean):
        self.time_resolution = time_resolution
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.resample(self.time_resolution).apply(self.func)


class stormsProcessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 storm_times_df=None,
                 storms_to_use=None,
                 start='1980',
                 end='2020',
                 time_resolution='5T',
                 vx_colname='vx_gse',
                 D=1500000,
                 min_threshold=None,
                 interpolate=True):
        self.storm_times_df = storm_times_df
        self.storms_to_use = storms_to_use
        self.start = start
        self.end = end
        self.time_resolution = time_resolution
        self.min_threshold = min_threshold
        self.interpolate = interpolate

    def fit(self, X, y=None, target_column='sym_h'):
        self.columns_ = X.columns
        self.times_ = X.index
        self.target_column_ = target_column

        if self.storm_times_df is None:
            self.data_ = X.to_numpy()
        else:
            # TODO: Error handling
            times_to_include = pd.date_range(start=self.start,
                                             end=self.end,
                                             freq=self.time_resolution,
                                             closed='left')
            self.storm_times_df = self.storm_times_df[
                (self.storm_times_df['start_time'] >= times_to_include[0]) &
                (self.storm_times_df['end_time'] <= times_to_include[-1])
            ]

            if self.storms_to_use is None:
                self.storms_to_use = self.storm_times_df.index
            else:
                self.storms_to_use = self.storm_times_df.index.intersection(self.storms_to_use)
            
            storm_indices = get_storm_indices(
                X, self.storm_times_df, self.storms_to_use, time_resolution=self.time_resolution)
            storm_indices_concat = np.concatenate(storm_indices)
            self.times_ = X.index[storm_indices_concat]

            if self.interpolate:
                processed_data = pd.concat([X.iloc[storm_indices[i]].interpolate(
                    method='time', axis=0, limit_direction='both').assign(storm=i) for i in range(len(storm_indices))])
            else:
                processed_data = pd.concat([X.iloc[storm_indices[i]].assign(
                    storm=i) for i in range(len(storm_indices))])

            self.storm_labels_ = processed_data['storm']
            self.data_ = processed_data
            # Remove storm column
            # self.data_ = np.delete(processed_data, -1, axis=1)

        return self

    def transform(self, X, y=None):
        X_ = self.data_.drop(columns=self.target_column_)
        y_ = self.data_[self.target_column_]
        return X_, y_

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

    def get_times(self):
        return self.times_

    def get_storm_labels(self):
        return self.storm_labels_

    def get_target_column(self):
        return self.target_column_

    # TODO: plot_storms function
    def plot_storms(self):
        pass

# TODO: Put interpolater here and allow user to specify it


class LagFeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 auto_order=10,
                 exog_order=10,
                 target_column=0,
                 label_column=-1):
        #  storm_labels=None):
        self.auto_order = auto_order
        self.exog_order = exog_order
        self.target_column = target_column
        self.label_column = label_column
        # self.storm_labels = storm_labels

    def fit(self, X, y=None):
        # Input validation
        # if self.target_column is None:
        #     raise ValueError("target_column must be specified.")
        _ = check_array(X)
        if self.label_column == self.target_column:
            raise ValueError("label_column cannot be the same as target_column.")
        # Save storm labels
        self.storm_labels_ = X[self.label_column]
        return self

    # IDEA: Let y contain target and get rid of target_column
    def transform(self, X, y=None):
        # check_is_fitted(self)
        if self.label_column is None:
            # if self.storm_labels_ is None:
            features = self._transform_one_storm(X)
        else:
            # TODO: Handle case when X is np array
            # unique_labels = np.unique(self.storm_labels_)
            # Remove label column
            # X = np.delete(X, self.label_column, axis=1)

            # Get lag features for each storm and combine
            # features = np.vstack([self._transform_one_storm(X, i)
            #                       for i in unique_labels])
            
            features = np.vstack(
                X.groupby(by=self.label_column).apply(self._transform_one_storm)
                )
            
        features = check_array(features, force_all_finite='allow-nan')
        return features

    def _transform_one_storm(self, X):
        y_ = X.iloc[:,self.target_column].to_numpy()
        X_ = X.drop(columns=[self.label_column, X.columns[self.target_column]]).to_numpy()
        
        # TODO: write my own version
        p = MetaLagFeatureProcessor(X_, y_, self.auto_order, [
                                    self.exog_order]*X_.shape[1], [0]*X_.shape[1])
        lagged_features = p.generate_lag_features()
        return lagged_features

    def get_storm_labels(self):
        check_is_fitted(self)
        return self.storm_labels_


class TargetProcessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 pred_step=0,
                 storm_labels=None,
                 propagated_times=None):
        self.pred_step = pred_step
        self.storm_labels = storm_labels
        self.propagated_times = propagated_times

    def fit(self, X, y=None):
        if self.propagated_times is not None:

            if X.shape[0] != len(self.propagated_times):
                raise ValueError(
                    "X.shape[0] must be equal to the length of propagated_times.")
            if X.index.duplicated().any():
                raise ValueError("X has duplicated indices.")
            if self.propagated_times.duplicated().any():
                raise ValueError("propagated_times has duplicate times.")
            
        return self

    def transform(self, X, y=None):
        return X.reindex(self.propagated_times).to_numpy()


class TargetProcessorPropagated(BaseEstimator, TransformerMixin):
    def __init__(self, propagated_times=None,
                 storm_labels=None):
        self.propagated_times = propagated_times
        self.storm_labels = storm_labels

    def fit(self, X, y=None):
        # Find where propagated_times have identical times
        zip(self.propagated_times, self.propagated_times[1:])

        return self

    def transform(self, X, y=None):
        pass

    def _transform_one_storm(self, X, storm_label=None):
        idx = np.where(self.storm_labels == storm_label)[0]
        # target =
