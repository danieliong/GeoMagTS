import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from cached_property import cached_property
from functools import wraps, partial

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, Lasso, LassoLars, LassoCV, LassoLarsCV
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array, check_consistent_length
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline
from pandas.tseries.frequencies import to_offset
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm
import joblib
import time

from GeoMagTS.models.processors import GeoMagARXProcessor, NotProcessedError, requires_processor_fitted
from GeoMagTS.utils import _get_NA_mask

# NOTE: Converting to pandas from np slowed down the code significantly

class GeoMagARXRegressor(RegressorMixin, BaseEstimator, GeoMagARXProcessor):
    def __init__(self,
                 base_estimator=None,
                 auto_order=60,
                 exog_order=60,
                 pred_step=0,
                 transformer_X=None,
                 transformer_y=None,
                 include_interactions=False,
                 interactions_degree=2,
                 seasonality=False,
                 propagate=True,
                 time_resolution='5T',
                 D=1500000,
                 storm_level=0,
                 time_level=1,
                 lazy=True,
                 verbose=False,
                 **estimator_params):
        super().__init__(
            exog_order=exog_order,
            pred_step=pred_step,
            transformer_X=transformer_X,
            transformer_y=transformer_y,
            include_interactions=include_interactions,
            interactions_degree=interactions_degree,
            seasonality=seasonality,
            propagate=propagate,
            time_resolution=time_resolution,
            D=D, storm_level=storm_level,
            time_level=time_level,
            lazy=lazy)
        
        self.verbose = verbose
        self.base_estimator = base_estimator.set_params(**estimator_params)
        self.estimator_params = estimator_params

    def _prefit(self, copy=True, **kwargs):

        processor_args = GeoMagARXProcessor.__init__.__code__.co_varnames[1:]
        # Set attributes
        for name, value in kwargs.items():
            if name in processor_args:
                setattr(self, name, value)
            else:
                warnings.warn(
                    name+" is not an argument in GeoMagARXProcessor. It will not be set as an attribute.")

        # Define or clone base estimator
        if self.base_estimator is None:
            # Default base estimator is LinearRegression
            self.base_estimator_fitted_ = LinearRegression()
        elif copy:
            self.base_estimator_fitted_ = clone(self.base_estimator)
        else:
            self.base_estimator_fitted_ = self.base_estimator

        return None

    def fit(self, X, y,
            storm_level=0,
            time_level=1,
            vx_colname='vx_gse', **fit_args):
        # Default fit method

        self._prefit(
            storm_level=storm_level,
            time_level=time_level,
            vx_colname=vx_colname,
        )

        if self.verbose:
            start_time = time.time()
            
        features, target = self.process_data(X, y, fit=True)
        
        if self.verbose:
            run_time = time.time() - start_time
            print("[Fit] Data processing took {:.0f} seconds.".format(run_time))

        if self.verbose:
            start_time = time.time()

        # Fit base estimator
        self.base_estimator_fitted_.fit(features, target, **fit_args)
        
        if self.verbose:
            run_time = time.time() - start_time
            print(
                "[Fit] Fitting "+self.base_estimator.__str__()+" took {:.0f} seconds.".format(run_time))

        return self

    def fit_cv(self, X, y, param_grid=None, 
               storm_level=0, time_level=1, 
               vx_colname='vx_gse', n_splits=5,
               save=False, load=False, 
               file_name='cv.pkl',
                **kwargs):
        
        if save is True and load is True:
            raise ValueError("save and load can't both be True.")
        
        if not load and param_grid is None:
            raise ValueError("param_grid must be specified if load is False.")
        
        self._prefit(
            storm_level=storm_level,
            time_level=time_level,
            vx_colname=vx_colname
        )
        features, target = self.process_data(X, y, fit=True)
        
        if load:
            self.gridsearch_cv_ = joblib.load(file_name)
        else:
            self.gridsearch_cv_ = GridSearchCV(
                self.base_estimator, param_grid=param_grid, 
                cv=GroupKFold(n_splits=n_splits), refit=False, **kwargs)
            self.gridsearch_cv_.fit(features, target, 
                                    groups=self.train_storms_)
            
        if save:
            joblib.dump(self.gridsearch_cv_, file_name)
            
        self.base_estimator_fitted_.set_params(
            **self.gridsearch_cv_.best_params_)
        features, target = self.process_data(X, y, fit=True)
        self.base_estimator_fitted_.fit(features, target)
        self.cv_is_fitted_ = True
        
        return self
    
    def load_cv(self, cv_file):
        self.gridsearch_cv_ = joblib.load(cv_file)
        self.base_estimator_fitted_.set_params(
            **self.gridsearch_cv_.best_params_)
        self.cv_is_fitted_ = True
        return None 
        
    def predict(self, X, y=None, **predict_params):
        # y can only be None if self.auto_order == 0
        check_is_fitted(self)

        if self.verbose:
            start_time = time.time()
            
        X_, y = self.process_data(X, y, fit=False,
                                  remove_NA=False)
        nan_mask = _get_NA_mask(X_)
        test_features = X_[nan_mask]
        
        if self.verbose:
            run_time = time.time() - start_time
            print("[Predict] Test data processing took {:.0f} seconds.".format(run_time))

        if self.verbose:
            start_time = time.time()

        ypred = self.base_estimator_fitted_.predict(
            test_features, **predict_params)
        
        if self.verbose:
            run_time = time.time() - start_time
            print(
                "[Predict] Predict took {:.0f} seconds.".format(run_time))
            
        if self.verbose:
            start_time = time.time()
        
        ypred = self.process_predictions(
            ypred, Vx=X[self.vx_colname][nan_mask])
        
        if self.verbose:
            run_time = time.time() - start_time
            print(
                "[Predict] Processing predictions took {:.0f} seconds.".format(run_time))

        return ypred

    def score(self, X, y, metric=mean_squared_error, 
              negative=True, squared=False, decimals=None):

        y_pred = self.predict(X, y)

        score = self.score_func(y, y_pred, metric=metric,
                                squared=squared)
        if decimals is not None:
            score = np.around(score, decimals=decimals)

        if negative:
            return -score
        else:
            return score

    def score_func(self, y, y_pred,
                   metric=mean_squared_error, **kwargs):
        y_ = y.reindex(y_pred.index)
        nan_mask = ~np.isnan(y_)

        return metric(y_[nan_mask], y_pred[nan_mask], **kwargs)

    def _more_tags(self):
        return {'allow_nan': True,
                'no_validation': True}

# estimator_checks = check_estimator(GeoMagTSRegressor())

def not_implemented_for_lasso(method):
    @wraps(method)
    def wrapped_method(self, *args, **kwargs):
        if self.lasso:
            method_name = method.__qualname__
            message = method_name+" has/have not been implemented for Lasso yet!"
            raise NotImplementedError(message)
        return method(self, *args, **kwargs)
    return wrapped_method


def lasso_method(method):
    @wraps(method)
    def wrapped_method(self, *args, **kwargs):
        if not self.lasso:
            method_name = method.__qualname__
            raise Exception(method_name+" is only available for Lasso.")
        return method(self, *args, **kwargs)
    return wrapped_method


class GeoMagLinearARX(GeoMagARXRegressor):
    def __init__(self, auto_order=10,
                 exog_order=10,
                 pred_step=1,
                 transformer_X=None,
                 transformer_y=None,
                 include_interactions=False,
                 interactions_degree=2,
                 lasso=False,
                 lars=False,
                 **kwargs):
        if lasso and lars:
            base_estimator = LassoLars()
        elif lasso and not lars:
            base_estimator = Lasso()
        else:
            base_estimator = LinearRegression()
        super().__init__(
            base_estimator=base_estimator,
            auto_order=auto_order,
            exog_order=exog_order,
            pred_step=pred_step,
            transformer_X=transformer_X,
            transformer_y=transformer_y,
            include_interactions=include_interactions,
            interactions_degree=interactions_degree,
            **kwargs)

        self.lasso = lasso
        self.lars = lars
        self.cv_is_fitted_ = False

    @property
    def coef_(self):
        return self.base_estimator_fitted_.coef_

    @cached_property
    @requires_processor_fitted
    def fitted_values_(self):
        fitted_vals = np.matmul(
            self.train_features_,
            self.coef_
        )
        return fitted_vals

    @cached_property
    @requires_processor_fitted
    def sigma_sq_(self):
        n, p = self.train_shape_
        diff = self.train_target_ - self.fitted_values_
        sigma_sq = diff.dot(diff) / (n - p)
        return sigma_sq

    @cached_property
    @requires_processor_fitted
    def inv_XTX_(self):
        inv_XTX = np.linalg.inv(
            np.matmul(
                self.train_features_.transpose(),
                self.train_features_
            )
        )
        return inv_XTX

    @cached_property
    @not_implemented_for_lasso
    def standard_error_(self, squared=False):

        mse = np.diag(self.sigma_sq_ * self.inv_XTX_)
        if squared:
            return mse
        else:
            return np.sqrt(mse)

    @cached_property
    @not_implemented_for_lasso
    @requires_processor_fitted
    def pvals_(self):

        from scipy.stats import t
        n, p = self.train_features_.shape
        df = n - p

        test_stat = np.abs(self.coef_) / self.standard_error_
        pval = np.around(2*t.sf(test_stat, df), decimals=3)
        return pval

    @cached_property
    @not_implemented_for_lasso
    def pval_df_(self):

        check_is_fitted(self)
        pval_df = self._format_df(self.pvals_)
        return pval_df

    @cached_property
    def coef_df_(self):
        check_is_fitted(self)
        coef_df = self._format_df(self.coef_)
        return coef_df
    
    @cached_property
    def train_errors(self):
        yhat = self.predict(self.train_features_)
        return (self.train_target_ - yhat)

    @not_implemented_for_lasso
    def compute_prediction_se(self, X, y=None, squared=False):

        test_features, y = self.process_data(X, y, fit=False,
                                             remove_NA=False)
        self.prediction_se_mask_ = _get_NA_mask(test_features)
        test_features = test_features[self.prediction_se_mask_]

        covar = self.sigma_sq_ * (
            test_features.dot(self.inv_XTX_.dot(test_features.transpose())) +
            np.eye(test_features.shape[0]))
        if squared:
            return np.diag(covar)
        else:
            return np.sqrt(np.diag(covar))

    @not_implemented_for_lasso
    def compute_prediction_interval(self, X, y=None, level=.95):

        from scipy.stats import t

        ypred = self.predict(X, y)
        self.prediction_se_ = self.compute_prediction_se(X, y)
        self.prediction_se_ = self.process_predictions(
            self.prediction_se_, Vx=X[self.vx_colname][self.prediction_se_mask_], inverse_transform_y=False
        )

        n, p = self.train_features_.shape

        lower_z, upper_z = t.interval(level, n-p)
        pred_interval = {'ypred': ypred,
                         'lower': ypred + (lower_z * self.prediction_se_),
                         'upper': ypred + (upper_z * self.prediction_se_)
                         }
        return pred_interval

    def _format_df(self, vals, decimals=3):
        ar_names = np.array(
            ["ar"+str(i) for i in range(self.auto_order_steps_)]
        )
        exog_names = np.concatenate(
            [[x+str(i) for i in range(self.exog_order_steps_)]
             for x in self.train_features_cols_]
        ).T

        names = np.concatenate([ar_names, exog_names])

        vals_no_interactions = pd.Series(vals[:len(names)], index=names)
        df = pd.DataFrame(
            {col: vals_no_interactions[vals_no_interactions.index.str.contains('^'+col+'[0-9]+$')].reset_index(drop=True)
             for col in self.train_features_cols_.insert(0, 'ar')
             }
        )

        if vals.shape[0] > len(names) and self.include_interactions:

            powers = self.interactions_processor_.powers_
            n_features = self.train_features_cols_.shape[0]
            colnames = self.train_features_cols_.insert(0, 'ar')

            interaction_masks = powers[n_features+1:].astype(bool)
            interaction_colnames = ['_'.join(colnames[mask].tolist())
                                    for mask in interaction_masks]
            interaction_names = np.concatenate(
                [[x+str(i) for i in range(self.exog_order_steps_)]
                 for x in interaction_colnames]
            )
            interactions = pd.Series(
                vals[len(names):len(names)+len(interaction_names)],
                index=interaction_names
            )
            interactions_df = pd.DataFrame(
                {col: interactions[interactions.index.str.contains('^'+col+'[0-9]+$')].reset_index(drop=True)
                 for col in interaction_colnames
                 }
            )
            df = pd.concat([df, interactions_df], axis=1)
            
        if self.seasonality:
            seasonality_names = ('sin_yr','cos_yr','sin_day','cos_day')
            seasonality_df = pd.DataFrame(
                {seasonality_names[i]: [vals[-i]]
                 for i in range(len(seasonality_names))}
                )
            df = pd.concat([df, seasonality_df], axis=1)

            # Set index to minutes lag
        df.set_index(
            np.arange(0, self.exog_order,
                      step=self.time_res_minutes_).astype(int),
            inplace=True)
        df.index.set_names('lag', inplace=True)

        if decimals is not None:
            df = df.round(decimals)

        return df

    @lasso_method
    def fit_cv(self, X, y, storm_level=0, time_level=1,
               vx_colname='vx_gse', n_splits=5, **cv_params):
        self._prefit(
            storm_level=storm_level,
            time_level=time_level,
            vx_colname=vx_colname
        )
        features, target = self.process_data(X, y, fit=True)

        if self.lars:
            self.base_estimator_fitted_ = LassoLarsCV()
        else:
            self.base_estimator_fitted_ = LassoCV()

        cv = GroupKFold(n_splits=n_splits)
        self.cv_split_ = list(
            cv.split(features, target, groups=self.train_storms_))

        self.base_estimator_fitted_.set_params(cv=self.cv_split_, **cv_params)
        self.base_estimator_fitted_.fit(features, target)

        self.cv_is_fitted_ = True

        return self

# class GeoMagGP(GeoMagTSRegressor):
#     def __init__(self, **params):
#         super().__init__(
#             base_estimator=GaussianProcessRegressor(),
#             **params
#             )
