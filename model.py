import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import safe_mask
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array, check_consistent_length
from sklearn.metrics.regression import mean_squared_error

from GeoMagTS.data_preprocessing import LagFeatureProcessor, TargetProcessor

class GeoMagTSRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 base_estimator=None,
                 auto_order=10,
                 exog_order=10,
                 pred_step=1,
                 #  storm_labels=None,
                 #  target_column=None,
                 transformer_X=None,  # Allow user to enter own regressor/transformer
                 transformer_y=None,
                 **estimator_params):
        # TODO: Allow base_estimator params to be passed in here.
        self.base_estimator = base_estimator.set_params(**estimator_params)
        
        self.auto_order = auto_order
        self.exog_order = exog_order
        self.pred_step = pred_step
        # self.storm_labels = storm_labels
        # self.target_column = target_column
        self.transformer_X = transformer_X
        self.transformer_y = transformer_y

    def fit(self, X, y, storm_labels=None, **fit_params):

        self.lag_feature_processor_ = LagFeatureProcessor(
            auto_order=self.auto_order,
            exog_order=self.exog_order,
            target_column=0,
            storm_labels=storm_labels)
        self.target_processor_ = TargetProcessor(
            pred_step=self.pred_step,
            storm_labels=storm_labels)

        X, y = check_X_y(X, y, y_numeric=True)
        if self.transformer_X is not None:
            X = self.transformer_X.fit_transform(X)
        if self.transformer_y is not None:
            y = self.transformer_y.fit_transform(y.reshape(-1, 1)).flatten()

        y_X_ = np.concatenate([y.reshape(-1, 1), X], axis=1)
        X_ = self.lag_feature_processor_.fit_transform(y_X_)
        y_ = self.target_processor_.fit_transform(y)

        # # IDEA: Put this in processors
        # # Checks that allow nan in X, y
        # X_ = check_array(X_, force_all_finite='allow-nan')
        # y_ = check_array(y_, force_all_finite='allow-nan', ensure_2d=False)

        # Remove NAs induced by LagFeatureProcessor, TargetProcessor
        target, features = self._removeNA(X_, y_)

        if self.base_estimator is None:
            # Default estimator is LinearRegression
            self.base_estimator_fitted_ = LinearRegression()
        else:
            self.base_estimator_fitted_ = clone(self.base_estimator)

        self.base_estimator_fitted_.fit(target, features, **fit_params)
        return self

    def predict(self, X, y, storm_labels=None):
        check_is_fitted(self)

        lag_feature_processor = LagFeatureProcessor(
            auto_order=self.auto_order,
            exog_order=self.exog_order,
            target_column=0,
            storm_labels=storm_labels)
        target_processor = TargetProcessor(
            pred_step=self.pred_step,
            storm_labels=storm_labels)

        X, y = check_X_y(X, y, y_numeric=True)
        if self.transformer_X is not None:
            X = self.transformer_X.transform(X)
        if self.transformer_y is not None:
            y = self.transformer_y.transform(y.reshape(-1, 1)).flatten()

        concat_data = np.concatenate([y.reshape(-1, 1), X], axis=1)
        y_X_ = lag_feature_processor.transform(concat_data)
        features = self._removeNA(y_X_)

        ypred = self.base_estimator_fitted_.predict(features)

        if self.transformer_y is not None:
            ypred = self.transformer_y.inverse_transform(ypred)

        return ypred

    def _removeNA(self, X, y=None):
        if y is None:
            mask = safe_mask(X, np.isnan(X).any(axis=1))
            return X[~mask]
        else:
            data_ = np.concatenate([y.reshape(-1, 1), X], axis=1)
            mask = np.isnan(data_).any(axis=1)
            # Check mask
            mask = safe_mask(X, mask)
            mask = safe_mask(y, mask)
            X_, y_ = check_X_y(X[~mask], y[~mask], y_numeric=True)
            return X_, y_

    def score(self, X, y, storm_labels=None, squared=True):
        # TODO: Error handling and more options
        y_pred = self.predict(X, y, storm_labels)
        mask = np.isnan(y) | np.isnan(y_pred)

        return mean_squared_error(y[~mask], y_pred[~mask], squared=squared)

    def plot_predict(self, X, y, storm_labels=None, times=None, display_info=False, **plot_params):
        y_pred = self.predict(X, y, storm_labels)
        # rmse = self.score(X, y, storm_labels, False)

        fig, ax = plt.subplots(sharex=True, figsize=(10, 7), **plot_params)
        if times is None:
            ax.plot(y[-len(y_pred):], label='Truth',
                    color='black', linewidth=0.5)
            ax.plot(y_pred,
                    label=str(self.pred_step)+'-step ahead prediction', color='red', linewidth=0.5)
        else:
            ax.plot(times, y[-len(y_pred):],
                    label='Truth', color='black', linewidth=0.5)
            ax.plot(times, y_pred,
                    label=str(self.pred_step)+'-step ahead prediction', color='red', linewidth=0.5)
        ax.legend()

        # TODO
        # if display_info:
        #     ax.set_title(
        #         'n_hidden='+str(self.base_estimator.get_params()['n_hidden'])+', ' +
        #         'learn_rate='+str(self.base_estimator.get_params()['learning_rate'])+', ' +
        #         'auto_order='+str(self.auto_order)+', ' +
        #         'exog_order='+str(self.exog_order[0])+', ' +
        #         'RMSE='+str(np.round(rmse, decimals=2)))

        return fig, ax

    def _more_tags(self):
        return {'allow_nan': True,
                'no_validation': True}

# estimator_checks = check_estimator(GeoMagTSRegressor())
