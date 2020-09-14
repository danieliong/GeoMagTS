from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from GeoMagTS.utils import MetaLagFeatureProcessor
import warnings

class GeoMagARXProcessor():
    def __init__(self,
                 auto_order=60,
                 exog_order=60,
                 pred_step=0,
                 transformer_X=None,
                 transformer_y=None,
                 propagate=True,
                 time_resolution='5T',
                 D=1500000):
        self.auto_order = auto_order
        self.exog_order = exog_order
        self.pred_step = pred_step
        self.transformer_X = transformer_X
        self.transformer_y = transformer_y
        self.propagate = propagate
        self.time_resolution = time_resolution
        self.D = D
        self.processor_fitted_ = False

    def check_data(self, X, y=None, fit=True, check_multi_index=True,
                   check_vx_col=True, check_same_cols=False, storm_level=None, time_level=None, **sklearn_check_params):

        # Check without converting to np arrays
        _ = check_X_y(X, y, y_numeric=True, **sklearn_check_params)

        if fit and y is None:
            return ValueError("y must be specified if fit is True.")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas object (for now).")

        if isinstance(X.index, pd.MultiIndex):
            if X.index.nlevels < 2:
                raise ValueError(
                    "X.index must have at least 2 levels corresponding to storm and times if it is a MultiIndex.")

            # Get mask for removing duplicates
            try:
                dupl_mask_ = X.index.get_level_values(
                    level=self.time_level_).duplicated(keep='last')
            except AttributeError:
                warnings.warn("self.time_level_ is not defined.")

                if time_level is not None:
                    warnings.warn(
                        "time_level is specified and will be used in removing duplicate times.")
                    dupl_mask_ = X.index.get_level_values(
                        level=time_level).duplicated(keep='last')
                else:
                    raise ValueError(
                        "Either self.time_level_ or time_level must be defined if X has a MultiIndex.")

        elif check_multi_index:
            raise TypeError("X must have a MultiIndex (for now).")

        elif isinstance(X.index, pd.DatetimeIndex):
            dupl_mask_ = X.index.duplicated(keep='last')

        elif not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("Time index must be of type pd.DatetimeIndex.")

        if fit:
            # Save dupl_mask_
            self.dupl_mask_ = dupl_mask_

        if dupl_mask_.any():
            warnings.warn(
                "Inputs have duplicated indices. Only one of each row with duplicated indices will be kept.")

        # Check if vx_colname is in X
        if check_vx_col and self.vx_colname_ not in X.columns:
            raise ValueError("X does not have column "+self.vx_colname_+".")

        if check_same_cols:
            if not X.columns.equals(self.features_):
                raise ValueError(
                    "X must have the same columns as the X used for training.")

        if y is not None:
            if not isinstance(y, (pd.Series, pd.DataFrame)):
                raise TypeError("y must be a pandas object (for now).")

            # Check if X, y have same times
            if not X.index.equals(y.index):
                raise ValueError("X, y do not have the same indices.")

            y = y[~dupl_mask_]

        X = X[~dupl_mask_]

        return X, y

    def process_data(self, X, y=None, fit=True, check_data=True,
                     remove_NA=True, **check_params):

        if check_data:
            X, y = self._check_data(X, y, fit=fit, **check_params)

        if y is not None:
            data_ = pd.concat([y, X], axis=1)
        elif self.auto_order == 0:
            data_ = X
        else:
            raise ValueError("y needs to be given if self.auto_order != 0.")

        if fit:
            # - Set feature processor as attribute and fit it
            # - Set target processor as attribute, fit and transform y

            # Fit feature processor
            self.feature_processor_ = FeatureProcessor(
                auto_order=self.auto_order,
                exog_order=self.exog_order,
                time_resolution=self.time_resolution,
                transformer=self.transformer_X)
            self.feature_processor_.fit(
                data_,
                target_column=0,
                storm_level=self.storm_level_,
                time_level=self.time_level_)

            # Fit target processor and transform training targets
            self.target_processor_ = TargetProcessor(
                pred_step=self.pred_step,
                time_resolution=self.time_resolution,
                transformer=self.transformer_y
            )
            y_ = self.target_processor_.fit_transform(
                y, storm_level=self.storm_level_, time_level=self.time_level_)

            # Fit propagation time processor and transform training targets
            if self.propagate:
                self.propagation_time_processor_ = PropagationTimeProcessor(
                    Vx=X[self.vx_colname_],
                    time_resolution=self.time_resolution,
                    D=self.D)
                y_ = self.propagation_time_processor_.fit_transform(
                    y_, time_level=self.time_level_)
            else:
                self.propagation_time_processor_ = None

            # BUG: lengths of X_, y_ differ when pred_step > 0

            self.processor_fitted_ = True

        elif not self.processor_fitted_:
            raise NotProcessedError(self)

        # Use feature_processor that was fit previously.
        X_ = self.feature_processor_.transform(data_)

        if fit:
            # Align X_ with y_
            target_mask = self.target_processor_.mask_
            if self.propagate:
                target_mask = np.logical_and(
                    target_mask, self.propagation_time_processor.mask_)
            X_ = X_[target_mask]

            if remove_NA:
                mask = _get_NA_mask(X_, y_)
                return X_[mask], y_[mask]
            else:
                return X_, y_
        else:
            if remove_NA:
                mask = _get_NA_mask(X_)
                return X_[mask]
            else:
                return X_

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
        self.transformer = transformer
        self.fit_transformer = fit_transformer
        self.transformer_params = transformer_params

    def fit(self, X, y=None, target_column=0, storm_level=0, time_level=1,
            **transformer_fit_params):

        if self.transformer is not None:
            self.transformer = self.transformer.set_params(**transformer_params)

        self.target_column_ = target_column
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
        # Check if time has regular increments
        if isinstance(X.index, pd.MultiIndex):
            times = X.index.get_level_values(level=self.time_level_)
        else:
            times = X.index
        # NOTE: This breaks down when we use 'min' instead of 'T' when
        # specifying time resolution. Fix later.
        if times.inferred_freq != self.time_resolution:
            raise ValueError(
                "X does not have regular time increments with the specified time resolution.")

        y_ = X.iloc[:, self.target_column_].to_numpy()
        X_ = X.drop(columns=X.columns[self.target_column_]).to_numpy()

        # TODO: write my own version
        p = MetaLagFeatureProcessor(
            X_, y_, self.auto_order_timesteps_,
            [self.exog_order_timesteps_]*X_.shape[1],
            [0]*X_.shape[1])
        lagged_features = p.generate_lag_features()
        return lagged_features

class PropagationTimeProcessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 Vx=None,
                 time_resolution='5T',
                 D=1500000):
        self.Vx = Vx
        self.time_resolution = time_resolution
        self.D = D

    def _compute_propagated_time(self, x):
        return x['time'] + pd.Timedelta(x['prop_time'], unit='sec').floor(freq=self.time_resolution)

    def _compute_times(self, storm_level=0, time_level=1):
        self.propagation_in_sec_ = self.D / self.Vx.abs()
        self.propagation_in_sec_.rename("prop_time", inplace=True)

        if isinstance(self.propagation_in_sec_.index, pd.MultiIndex):
            self.propagation_in_sec_.index.rename(
                names='time', level=time_level, inplace=True)
            self.propagated_times_ = self.propagation_in_sec_.reset_index(
                level=time_level).apply(self._compute_propagated_time, axis=1)
        else:
            self.propagation_in_sec_.index.rename('time', inplace=True)
            self.propagated_times_ = self.propagation_in_sec_.reset_index().apply(
                self._compute_propagated_time, axis=1)
            self.propagated_times_.rename('times', inplace=True)

        return None

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

    def fit(self, X, y=None, storm_level=0, time_level=1):

        if self.Vx is None:
            raise ValueError("Vx must be specified.")
        elif not isinstance(self.Vx, pd.Series):
            raise TypeError("Vx must be a pd.Series (for now).")

        self.storm_level_ = storm_level
        self.time_level_ = time_level

        if X.shape[0] != self.Vx.shape[0]:
            # Reindex Vx to have same index as X
            self.Vx = self.Vx.reindex(X.index)

        self._compute_times(storm_level=storm_level,
                            time_level=time_level)
        self.mask_ = self._get_mask(X)

        return self

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

class NotProcessedError(Exception):
    def __init__(self, obj):
        self.class_name = obj.__class__.__name__
        self.message = "Data have not been previously processed using process_data in " + \
            self.class_name+". Please call"+self.class_name + \
            ".process_data with fit=True first."
