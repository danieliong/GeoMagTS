from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.utils import safe_mask, check_scalar
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array, check_consistent_length

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
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

    def _process_one_storm(self, X, storm_idx):
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
                [self._process_one_storm(X, i)
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

# Returns df with 3rd index = propagated_time


class PropagationTimeProcessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 Vx=None,
                 time_resolution='5T',
                 D=1500000):
        self.Vx = Vx
        self.time_resolution = time_resolution
        self.D = D

    def fit(self, X, y=None, storm_level=0, time_level=1):

        if self.Vx is not None and not isinstance(self.Vx, pd.Series):
            raise TypeError("Vx must be a pd.Series (for now).")

        self.storm_level_ = storm_level
        self.time_level_ = time_level
        
        if X.shape[0] != self.Vx.shape[0]:
            # Reindex Vx to have same index as X
            self.Vx = self.Vx.reindex(X.index)

        if self.Vx is not None:
            self.propagation_in_sec_ = self.D / self.Vx.abs()
            self.propagation_in_sec_.rename("prop_time", inplace=True)

            if isinstance(self.propagation_in_sec_.index, pd.MultiIndex):
                self.propagation_in_sec_.index.rename(names='time', level=time_level, inplace=True)
                self.propagated_times_ = self.propagation_in_sec_.reset_index(
                    level=time_level).apply(self._compute_propagated_time, axis=1)
            else:
                self.propagation_in_sec_.index.rename('time', inplace=True)
                self.propagated_times_ = self.propagation_in_sec_.reset_index().apply(self._compute_propagated_time, axis=1)
                
            self.mask_ = self._get_mask(X)
            
        return self

    def _compute_propagated_time(self, x):
        return x['time'] + pd.Timedelta(x['prop_time'], unit='sec').floor(freq=self.time_resolution)

    def _get_mask(self, X):
        # Get mask of where propagated times are in X's time index
        if isinstance(X.index, pd.MultiIndex):
            X_times = X.index.get_level_values(level=self.time_level_)
            proptime_in_X_mask = np.in1d(self.propagated_times_, X_times)
        else:
            proptime_in_X_mask = np.in1d(self.propagated_times_, X.index)
            
        # Get mask of where propagated times are duplicated
        dupl_mask = self.propagated_times_.duplicated(keep='last').values
        mask = np.logical_and(proptime_in_X_mask, ~dupl_mask)
        return mask

    def transform(self, X, y=None):
        check_is_fitted(self)

        if self.Vx is not None:
            if isinstance(X.index, pd.MultiIndex):
                storms = X.index.get_level_values(level=self.storm_level_)
                return X.reindex(
                    [storms[self.mask_], self.propagated_times_[self.mask_]]
                    )
            else:
                return X.reindex(self.propagated_times_[self.mask_])
        else:
            return X


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


class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 auto_order=60,
                 exog_order=60,
                 time_resolution='5T',
                 transformer=None,
                 fit_transformer=True,
                 **transformer_params):
        self.auto_order = auto_order
        self.exog_order = exog_order
        self.time_resolution = time_resolution
        self.transformer = transformer.set_params(**transformer_params)
        self.fit_transformer = fit_transformer
        self.transformer_params = transformer_params

    def fit(self, X, y=None, target_column=0, storm_level=0, time_level=1,
            **transformer_fit_params):
        self.target_column_ = X.columns[target_column]
        self.storm_level_ = storm_level
        self.time_level_ = time_level

        # convert auto_order, exog_order to time steps
        time_res_minutes = to_offset(self.time_resolution).delta.seconds / 60
        self.auto_order_timesteps_ = np.rint(
            self.auto_order / time_res_minutes).astype(int)
        self.exog_order_timesteps_ = np.rint(
            self.exog_order / time_res_minutes).astype(int)

        if self.transformer is not None:
            self.transformer.set_params(**self.transformer_params)

            if self.fit_transformer:
                self.transformer = _pd_fit(
                    self.transformer, X, **transformer_fit_params)

        return self

    def transform(self, X, y=None):
        check_is_fitted(self)

        X = _pd_transform(self.transformer, X)
        if isinstance(X.index, pd.MultiIndex):
            features = X.groupby(level=self.storm_level_).apply(
                self._transform_one_storm)
            features = np.vstack(features)
        else:
            features = self._transform_one_storm(X)
        
        features = check_array(features, force_all_finite='allow-nan')
        return features

    def _transform_one_storm(self, X):
        y_ = X[self.target_column_].to_numpy()
        X_ = X.drop(columns=self.target_column_).to_numpy()

        # TODO: write my own version
        p = MetaLagFeatureProcessor(
            X_, y_, self.auto_order_timesteps_,
            [self.exog_order_timesteps_]*X_.shape[1],
            [0]*X_.shape[1])
        lagged_features = p.generate_lag_features()
        return lagged_features
        

class TargetProcessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 pred_step=0,
                 time_resolution='5T',
                 #  propagate=True,
                 #  Vx=None,
                 #  D=1500000,
                 transformer=None,
                 **transformer_params):
        self.pred_step = pred_step
        self.time_resolution = time_resolution
        # self.propagate = propagate
        # self.Vx = Vx
        # self.D = D
        self.transformer = transformer
        self.transformer_params = transformer_params

    def fit(self, X, y=None, storm_level=0, time_level=1,
            **transformer_fit_params):

        self.storm_level_ = storm_level
        self.time_level_ = time_level

        # Transform X if transformer is provided
        if self.transformer is not None:
            self.transformer.set_params(**self.transformer_params)
            self.transformer = _pd_fit(
                self.transformer, X, **transformer_fit_params)

        # Get future times to predict
        self.times_to_predict_ = X.index.get_level_values(
            level=self.time_level_) + pd.Timedelta(minutes=self.pred_step)
        self.mask_ = self._get_mask(X)
        
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)

        X_ = _pd_transform(self.transformer, X)
        
        storms = X.index.get_level_values(level=self.storm_level_)
        X_ = X.reindex(
            [storms, self.times_to_predict_]
            )

        return X_                

    def _get_mask(self, X):
        X_times = X.index.get_level_values(level=self.time_level_)

        # Get mask of where times_to_predict are in X's time index
        mask = np.in1d(self.times_to_predict_, X_times)

        return mask


def _pd_fit(transformer, X, y=None, **transformer_fit_params):
    if transformer is not None:
        if isinstance(X, pd.Series):
            X_np = X.to_numpy().reshape(-1, 1)
            return transformer.fit(X_np)
        elif isinstance(X, pd.DataFrame):
            return transformer.fit(X, **transformer_fit_params)
        else:
            raise TypeError(
                "_pd_fit is meant to only take in pandas objects as input.")
    else:
        return None


def _pd_transform(transformer, X, y=None):
    if transformer is not None:
        check_is_fitted(transformer)
        X_transf = transformer.transform(X)
        X_df = pd.DataFrame(X_transf, index=X.index,
                            columns=X.columns).astype(X.dtypes)
        return X_df
    else:
        return X
