import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.utils import safe_mask
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array, check_consistent_length
from sklearn.metrics import mean_squared_error

from GeoMagTS.data_preprocessing import LagFeatureProcessor, TargetProcessor
from GeoMagTS.utils import _get_NA_mask

class GeoMagTSRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 base_estimator=None,
                 auto_order=10,
                 exog_order=10,
                 pred_step=0,
                 #  storm_labels=None,
                 #  target_column=None,
                 transformer_X=None,
                 transformer_y=None,
                 vx_colname='vx_gse',
                 D=1500000,
                 propagated_times = None, # TODO: Remove later 
                 **estimator_params):
        self.base_estimator = base_estimator.set_params(**estimator_params)
        
        self.auto_order = auto_order
        self.exog_order = exog_order
        self.pred_step = pred_step
        # self.storm_labels = storm_labels
        # self.target_column = target_column
        self.transformer_X = transformer_X
        self.transformer_y = transformer_y
        # self.propagated_times = propagated_times
        self.vx_colname = vx_colname
        self.D = D
        
    def _remove_duplicates(self, X, y):
        # Remove duplicated indices in X, y
        idx_dupl = X.index.duplicated()
        # Remove duplicated times in propagated_times
        prop_time_dupl_ = self.propagated_times.duplicated(
            keep='last').values
        
        mask = np.logical_or(idx_dupl, prop_time_dupl_)
        self.propagated_times_ = self.propagated_times[~mask]
        return X[~mask], y[~mask]
    
    # INCOMPLETE 
    def _compute_propagated_times(self, X, 
                                  vx_colname, D):
        
        def _compute_propagated_time(x):
            return x.name + pd.Timedelta(x['prop_times'], unit='sec')
        # TODO: Get frequency of X 
        
        times_df_ = X[vx_colname].to_frame()
        times_df_['prop_times'] = D / times_df_[vx_colname].abs()
        propagated_times = times_df_.apply(_compute_propagated_time)    

    def fit(self, X, y, label_column='storm', **fit_args):
        # Input validation
        _ = check_X_y(X, y, y_numeric=True)
        
        # Check if X, y have same times
        if not X.index.equals(y.index):
            raise ValueError("X, y do not have the same indices.")
        
        X, y = self._remove_duplicates(X, y)
        
        # INCOMPLETE 
        # propagated_times = self._compute_propagated_times(
        #     X=X, vx_colname=self.vx_colname, D=self.D)
        
        ### Process features
        # IDEA: Write function for this. predict does something similar
        self.lag_feature_processor_ = LagFeatureProcessor(
            auto_order=self.auto_order,
            exog_order=self.exog_order,
            target_column=0,
            label_column=label_column)
        if self.transformer_X is None:
            concat_data = pd.concat([y, X], axis=1)
        else:
            self.transformer_X.fit(X.drop(columns=label_column))
            X_transf = self._transform_X_pd(X, label_column)
            concat_data = pd.concat([y, X_transf], axis=1)
                       
        X_ = self.lag_feature_processor_.fit_transform(concat_data)
        storm_labels = self.lag_feature_processor_.get_storm_labels()
        
        ### Process target 
        self.target_processor_ = TargetProcessor(
            pred_step=self.pred_step,
            storm_labels=storm_labels,
            propagated_times=self.propagated_times_)
        if self.transformer_y is not None:
            y_np = y.to_numpy()
            y_np = self.transformer_y.fit_transform(y_np.reshape(-1, 1)).flatten()
            y = pd.Series(y_np, index=y.index)
        y_ = self.target_processor_.fit_transform(y)
        
        # Remove NAs induced by LagFeatureProcessor, TargetProcessor
        X_y_mask = _get_NA_mask(X_, y_)
        target, features = X_[~X_y_mask], y_[~X_y_mask]

        if self.base_estimator is None:
            # Default estimator is LinearRegression
            print("Default base estimator is LinearRegression.")
            self.base_estimator_fitted_ = LinearRegression()
        else:
            self.base_estimator_fitted_ = clone(self.base_estimator)

        self.base_estimator_fitted_.fit(target, features, **fit_args)
        return self    

    # Hacky way of getting transformer to output pd DataFrame
    def _transform_X_pd(self, X, label_column=None):
        X_transf = self.transformer_X.transform(X.drop(columns=label_column))
        X_transf = pd.DataFrame(X_transf, 
                                index=X.index,
                                columns=X.columns.drop(label_column))
        X_transf[label_column] = X[label_column]
        return X_transf

    def predict(self, X, y, label_column='storm'):
        check_is_fitted(self)

        lag_feature_processor_ = LagFeatureProcessor(
            auto_order=self.auto_order,
            exog_order=self.exog_order,
            target_column=0,
            label_column=label_column)
        if self.transformer_X is None:
            concat_data = pd.concat([y, X], axis=1)
        else:
            # Mask for removing label_column
            # X_mask = [True]*X.shape[1]
            # X_mask[label_column] = False
            # mask = [True] + X_mask
            # Transform data
            self.transformer_X.fit(X.drop(columns=label_column))
            X_transf = self._transform_X_pd(X, label_column)
            concat_data = pd.concat([y, X_transf], axis=1)
        X_ = lag_feature_processor_.fit_transform(concat_data)
        X_mask = _get_NA_mask(X_)
        features = X_[~X_mask]

        y_pred_ahead = self.base_estimator_fitted_.predict(features)
        if self.transformer_y is not None:
            y_pred_ahead = self.transformer_y.inverse_transform(y_pred_ahead.reshape(-1,1)).flatten()

        y_pred = np.empty(y.shape[0])
        y_pred[X_mask], y_pred[~X_mask] = np.nan, y_pred_ahead  
        y_pred_aligned = np.concatenate(
            [np.empty(self.pred_step)*np.nan, y_pred]
            )[0:len(y)]
 
        return y_pred_aligned

    def plot_predict(self, X, y, times=None, display_info=False, **plot_params):
        y_pred = self.predict(X, y)
        rmse = self.score(X, y, squared=False)

        fig, ax = plt.subplots(sharex=True, figsize=(10, 7), **plot_params)
        if times is None:
            ax.plot(y, label='Truth',
                    color='black', linewidth=0.5)
            ax.plot(y_pred,
                    label=str(self.pred_step)+'-step ahead prediction', color='red', linewidth=0.5)
        else:
            # TODO: Check if this subsetting is valid. Find better to do it?
            ax.plot(times, y,
                    label='Truth', color='black', linewidth=0.5)
            ax.plot(times, y_pred,
                    label=str(self.pred_step)+'-step ahead prediction', color='red', linewidth=0.5)
        ax.legend()
        locator = mdates.AutoDateLocator(minticks=15)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        # TODO
        if display_info:    
            ax.set_title(
                'auto_order='+str(self.auto_order)+', ' +
                'exog_order='+str(self.exog_order)+', ' +
                'RMSE='+str(np.round(rmse, decimals=2)))

        return fig, ax

    def score(self, X, y, squared=False):
        # TODO: Error handling and more options
        y_pred = self.predict(X, y)
        mask = np.isnan(y) | np.isnan(y_pred)

        return mean_squared_error(y[~mask], y_pred[~mask], squared=squared)


    def _more_tags(self):
        return {'allow_nan': True,
                'no_validation': True}

# estimator_checks = check_estimator(GeoMagTSRegressor())

class GeoMagARX(GeoMagTSRegressor):
    def __init__(self, auto_order=10, 
                 exog_order=10, 
                 pred_step=1, 
                 transformer_X=None, 
                 transformer_y=None, 
                 **lm_params):
        base_estimator = LinearRegression()
        super().__init__(base_estimator=base_estimator, auto_order=auto_order, exog_order=exog_order, pred_step=pred_step, transformer_X=transformer_X, transformer_y=transformer_y, **lm_params)
    
    def get_coef_df(self, feature_columns):
        check_is_fitted(self)
        ar_coef_names = np.array(
            ["ar"+str(i) for i in range(self.auto_order)]
            )
        exog_coef_names = np.concatenate(
            [[x+str(i) for i in range(self.exog_order)] 
             for x in feature_columns]
            ).T
        coef_names = np.concatenate([ar_coef_names, exog_coef_names])
        
        coef = pd.Series(
            self.base_estimator_fitted_.coef_, index=coef_names)
        coef_df = pd.DataFrame(
            {col: coef[coef.index.str.contains(col+'[0-9]+$')].reset_index(drop=True)
             for col in ['ar']+feature_columns
             }
            )
        coef_df['ar'] = coef_df['ar'].shift(1)
              
        return coef_df
    
class PersistenceModel(BaseEstimator, RegressorMixin):
    def __init__(self, pred_step=1,
                 transformer_X=None, 
                 transformer_y=None):
        self.pred_step = pred_step
        self.transformer_X = transformer_X
        self.transformer_y = transformer_y
    
    
      
