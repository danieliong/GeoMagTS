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
                 interpolate=True):
        self.storm_times_df = storm_times_df
        self.storms_to_use = storms_to_use
        self.storms_to_delete = storms_to_delete
        self.start = start
        self.end = end
        self.target_column = target_column
        self.time_resolution = time_resolution
        self.min_threshold = min_threshold
        self.interpolate = interpolate

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

    def _process_one_storm(self, X, storm_idx):
        X_one_storm = X.iloc[self.storm_indices_[
            storm_idx]].assign(storm=storm_idx)

        if self.interpolate:
            X_one_storm = X_one_storm.interpolate(
                method='time', axis=0, limit_direction='both')

        return X_one_storm

    def transform(self, X, y=None):
        if self.storm_times_df is not None:
            X_processed = pd.concat(
                [self._process_one_storm(X, i)
                 for i in range(len(self.storm_indices_))]
            )
            X_processed.reset_index(inplace=True)
            X_processed.set_index(['storm', 'times'], inplace=True)

            return X_processed
        else:
            return X

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
                    self.storms_thresholded, self.test_size)

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
        y = X[self.target_column]
        if self.test_storms is None:
            test_idx = pd.IndexSlice[self.test_storms_, :]
        else:
            test_idx = pd.IndexSlice[self.test_storms, :]
        train_idx = pd.IndexSlice[self.train_storms_, :]

        # data = {
        #     'X_train': X.iloc[train_idx],
        #     'y_train': y.iloc[train_idx],
        #     'X_test': X.iloc[test_idx],
        #     'y_test': y.iloc[test_idx]
        # }

        X_train, y_train = X.loc[train_idx, :], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx, :], y.loc[test_idx]
        return X_train, y_train, X_test, y_test

    def get_train_test_storms(self):
        check_is_fitted(self)
        if self.test_storms is None:
            return self.train_storms_, self.test_storms_
        else:
            return self.train_storms, self.test_storms_

class TimeShifter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

class PandasConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

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
        # self.label_column = label_column
        # self.storm_labels = storm_labels

    def fit(self, X, y=None):
        # Input validation
        # if self.target_column is None:
        #     raise ValueError("target_column must be specified.")
        _ = check_array(X)
        # if self.label_column == self.target_column:
        #     raise ValueError(
        #         "label_column cannot be the same as target_column.")
        # Save storm labels
        # self.storm_labels_ = X[self.label_column]
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
        y_ = X.iloc[:, self.target_column].to_numpy()
        X_ = X.drop(columns=[self.label_column,
                             X.columns[self.target_column]]).to_numpy()

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
