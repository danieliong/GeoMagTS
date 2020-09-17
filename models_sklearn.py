import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.utils import safe_mask
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array, check_consistent_length
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline
from pandas.tseries.frequencies import to_offset
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm

from GeoMagTS.processors import GeoMagARXProcessor, NotProcessedError
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
                 propagate=True,
                 time_resolution='5T',
                 D=1500000, 
                 storm_level=0,
                 time_level=1,
                 save_lazy_data=True,
                 **estimator_params):
        super().__init__(
            exog_order=exog_order,
            pred_step=pred_step,
            transformer_X=None,
            transformer_y=None,
            propagate=True,
            time_resolution=time_resolution,
            D=D, storm_level=storm_level,
            time_level=time_level,
            save_lazy_data=save_lazy_data)
        
        self.base_estimator = base_estimator.set_params(**estimator_params)
        self.estimator_params = estimator_params
    
    def fit(self, X, y,
            storm_level=0,
            time_level=1,
            vx_colname='vx_gse', **fit_args):

        self.storm_level_ = storm_level
        self.time_level_ = time_level
        self.vx_colname_ = vx_colname
        self.features_ = X.columns

        # Can't do this if we want to tune base_estimator params in GridSearchCV
        # self.base_estimator = self.base_estimator.set_params(
        #     **self.estimator_params)

        features, target = self.process_data(X, y, fit=True)
        
        # Get number of auto order, exog order steps
        time_res_minutes = to_offset(self.time_resolution).delta.seconds/60
        self.auto_order_steps_ = np.rint(
            self.auto_order / time_res_minutes).astype(int)
        self.exog_order_steps_ = np.rint(
            self.exog_order / time_res_minutes).astype(int)

        if self.base_estimator is None:
            # Default estimator is LinearRegression
            print("Default base estimator is LinearRegression.")
            self.base_estimator_fitted_ = LinearRegression()
        else:
            self.base_estimator_fitted_ = clone(self.base_estimator)

        # Fit base estimator 
        self.base_estimator_fitted_.fit(features, target, **fit_args)
        
        return self
    
    def predict(self, X, y=None, **predict_params):
        # y can only be None if self.auto_order == 0
        
        check_is_fitted(self)
        
        # X, y = self.check_data(X, y, fit=False, check_multi_index=False, check_vx_col=False, check_same_cols=True)
        
        X_, y = self.process_data(X, y, fit=False, 
                                  remove_NA=False)

        nan_mask = _get_NA_mask(X_)
        test_features = X_[nan_mask]
        
        ypred = self.base_estimator_fitted_.predict(
            test_features, **predict_params)
        
        ypred = self.process_predictions(
            ypred, Vx=X[self.vx_colname_][nan_mask])

        return ypred

    def score(self, X, y, negative=True, persistence=False, squared=False, 
              round=False, **round_params):
        
        if persistence:
            y_pred = self._predict_persistence(X, y)
        else:
            y_pred = self.predict(X, y)

        score = self.score_func(y, y_pred, squared=squared)
        if round:
            score = np.around(score, **round_params)

        if negative:
            return -score
        else:
            return score
    
    def score_func(self, y, y_pred, **kwargs):
        y_ = y.reindex(y_pred.index)
        nan_mask = ~np.isnan(y_)
        
        return mean_squared_error(y_[nan_mask], y_pred[nan_mask], **kwargs)

    def _more_tags(self):
        return {'allow_nan': True,
                'no_validation': True}

# estimator_checks = check_estimator(GeoMagTSRegressor())


class GeoMagLinearARX(GeoMagARXRegressor):
    def __init__(self, auto_order=10,
                 exog_order=10,
                 pred_step=1,
                 transformer_X=None,
                 transformer_y=None,
                 lasso=False,
                 **params):
        if lasso:
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
            **params)

    def compute_prediction_se(self, X, y=None, squared=False):
        
        if not self.processor_fitted_:
            raise NotProcessedError(self)
        elif not self.save_lazy_data:
            raise ValueError("self.save_lazy_data must be True to compute prediction standard errors.")
        
        test_features, y = self.process_data(X, y, fit=False,
                                             remove_NA=False)

        nan_mask = _get_NA_mask(test_features)
        test_features = test_features[nan_mask]
        
        n,p = self.train_features_.shape
        fitted_vals = self.compute_fitted_values()
        diff = self.train_target_.evaluate() - fitted_vals
        sigma_sq = diff.dot(diff) / (n - p)
        inv_XTX = np.linalg.inv(
            np.matmul(
                self.train_features_.evaluate().transpose(),
                self.train_features_.evaluate()
            )
        )
        
        covar = sigma_sq * (
            test_features.dot(inv_XTX.dot(test_features.transpose())) + \
            np.eye(test_features.shape[0]))
        if squared:
            return np.diag(covar)
        else:
            return np.sqrt(np.diag(covar)) 
    
    def compute_fitted_values(self):
        if not self.processor_fitted_:
            raise NotProcessedError(self)
        elif not self.save_lazy_data:
            raise ValueError(
                "self.save_lazy_data must be True to compute fitted values.")
        
        coef = self.base_estimator_fitted_.coef_
        fitted_vals = np.matmul(
            self.train_features_.evaluate(), 
            self.base_estimator_fitted_.coef_
            )
        return fitted_vals
    
    def compute_prediction_interval(self, X, y=None, level=.95):
        
        ypred = self.predict(X, y)
        se = self.compute_prediction_se(X, y)
    
        lower_z, upper_z = norm.interval(level)
        res = {'mean': ypred, 
               'lower': ypred + (lower_z * se),
               'upper': ypred + (upper_z * se)
               }
        return res 
        

    def get_coef_df(self, include_interactions=False):
        
        check_is_fitted(self)
        
        ar_coef_names = np.array(
            ["ar"+str(i) for i in range(self.auto_order_steps_)]
        )
        exog_coef_names = np.concatenate(
            [[x+str(i) for i in range(self.exog_order_steps_)]
             for x in self.features_]
        ).T
        
        coef_names = np.concatenate([ar_coef_names, exog_coef_names])

        coef = pd.Series(
            self.base_estimator_fitted_.coef_[0:len(coef_names)], 
            index=coef_names)
        coef_df = pd.DataFrame(
            {col: coef[coef.index.str.contains('^'+col+'[0-9]+$')].reset_index(drop=True)
             for col in self.features_.insert(0,'ar')
             }
        )
        
        if include_interactions:
            
            if isinstance(self.transformer_X, Pipeline):
                transformers = [step[0] for step in self.transformer_X.steps]
                if not np.isin(transformers, 'polynomialfeatures').any():
                    raise ValueError("Interaction terms were not computed.")
                powers = self.transformer_X['polynomialfeatures'].powers_
            else:
                if not isinstance(self.transformer_X, PolynomialFeatures):
                    raise ValueError("Interaction terms were not computed.")
                powers = self.transformer_X.powers_
            
            n_features = self.features_.shape[0]
            colnames = self.features_.insert(0, 'ar')
            interaction_masks = powers[n_features+1:].astype(bool)
            interaction_colnames = ['_'.join(colnames[mask].tolist())
                                 for mask in interaction_masks]
            interaction_names = np.concatenate(
                [[x+str(i) for i in range(self.exog_order_steps_)]
                 for x in interaction_colnames]
            )
            interactions = pd.Series(
                self.base_estimator_fitted_.coef_[len(coef_names):],
                index=interaction_names
            )
            interactions_df = pd.DataFrame(
                {col: interactions[interactions.index.str.contains('^'+col+'[0-9]+$')].reset_index(drop=True)
                 for col in interaction_colnames
                 }
            )
            coef_df = pd.concat([coef_df, interactions_df], axis=1)  

        return coef_df

# class GeoMagGP(GeoMagTSRegressor):
#     def __init__(self, **params):
#         super().__init__(
#             base_estimator=GaussianProcessRegressor(),
#             **params
#             )
        
        
