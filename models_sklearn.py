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

from GeoMagTS.processors import GeoMagARXProcessor
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
                 D=1500000, **estimator_params):
        super().__init__(
            exog_order=60,
            pred_step=0,
            transformer_X=None,
            transformer_y=None,
            propagate=True,
            time_resolution='5T',
            D=1500000)
        
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

        target, features = self.process_data(X, y, fit=True)
        
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
        self.base_estimator_fitted_.fit(target, features, **fit_args)
        
        return self
    
    def predict(self, X, y=None, **predict_params):
        # y can only be None if self.auto_order == 0
        
        check_is_fitted(self)
        
        X, y = self.check_data(X, y, fit=False, check_multi_index=False, check_vx_col=False, check_same_cols=True)
        
        X_, y = self.process_data(X, y, fit=False, 
                                  check_data=False,
                                  remove_NA=False)

        nan_mask = _get_NA_mask(X_)
        test_features = X_[nan_mask]
        
        ypred = self.base_estimator_fitted_.predict(
            test_features, **predict_params)
        
        ypred = self.process_predictions(
            ypred, Vx=X[self.vx_colname_][nan_mask])

        return ypred
    
    def _predict_persistence(self, X, y):
        y_pred = y.shift(periods=self.pred_step, freq='T')
        if self.propagate:
            pers_prop_time_processor = PropagationTimeProcessor(
                Vx=X[self.vx_colname_], time_resolution=self.time_resolution,
                D=self.D)
            pers_prop_time_processor = pers_prop_time_processor.fit(y_pred)
            y_pred = pd.Series(y_pred.values, 
                index = pers_prop_time_processor.propagated_times_.values)
            
            # y_pred = pers_prop_time_processor.shift(y_pred)
            
        return y_pred

    def _plot_one_storm(
            self, X, y, storm_idx, display_info=False,
            figsize=(10, 7), plot_persistence=True, model_name=None, 
            sw_to_plot=None, **plot_params):

        if sw_to_plot is not None:
            n_sw_to_plot = len(sw_to_plot)
            fig, ax = plt.subplots(nrows=n_sw_to_plot+1, 
                                   ncols=1,
                                   sharex=True,
                                   figsize=figsize,
                                   gridspec_kw={
                                       'height_ratios':[4]+[1]*n_sw_to_plot},
                                   **plot_params)
            ax0 = ax[0]
        else:
            fig, ax = plt.subplots(sharex=True,
                                   figsize=figsize,
                                   **plot_params)
            ax0 = ax
        
        ### Plot truth
        ax0.plot(y, label='Truth', color='black', linewidth=0.5)
        
        ### Plot predictions
        ypred = self.predict(X, y)
        rmse = self.score(X, y, negative=False, round=True, decimals=2)
    
        # Get prediction label
        pred_label = ''
        if self.propagate:
            pred_label = pred_label + 'Propagated '
        if self.pred_step > 0:
            pred_label = pred_label + str(self.pred_step) + 'min. ahead'
        pred_label = pred_label + 'prediction (RMSE: '+ str(rmse)+')'
        
        ax0.plot(ypred, label=pred_label, color='red', linewidth=0.5)

        if plot_persistence:
            y_persistence = self._predict_persistence(X, y)
            rmse_persistence = self.score(
                X, y, negative=False, persistence=True, squared=False, round=True, decimals=2)
            
            persistence_label = 'Persistence (RMSE: '+str(rmse_persistence)+')'
            ax0.plot(y_persistence, label=persistence_label,
                    color='blue', linestyle='--', linewidth=0.5)

        ax0.legend()
        # Adjust time scale
        locator = mdates.AutoDateLocator(minticks=15)
        formatter = mdates.ConciseDateFormatter(locator)
        ax0.xaxis.set_major_locator(locator)
        ax0.xaxis.set_major_formatter(formatter)

        if display_info:
            info = 'Storm #'+str(storm_idx)+": " + \
            'time_resolution='+str(self.time_resolution)+', '+ \
            'auto_order='+str(self.auto_order)+', ' + \
            'exog_order='+str(self.exog_order)+' (in min.)'
            if model_name is not None:
                info = info + ', ' + model_name 
            
            if len(self.estimator_params) != 0:
                info = info + ' ('
                i = 0
                for param, value in self.estimator_params.items():
                    info = info + param + '=' + str(value)
                    if i != len(self.estimator_params)-1:
                        info = info + ', '
                        i = i + 1
                info = info + ')'
            ax0.set_title(info)
        
        if sw_to_plot is not None:
            
            for i in range(n_sw_to_plot):
                ax[i+1].plot(X[sw_to_plot[i]], label=sw_to_plot[i], 
                        color='black', linewidth=0.5)
                ax[i+1].legend()
                ax[i+1].xaxis.set_major_locator(locator)
                ax[i+1].xaxis.set_major_formatter(formatter)
                       
            fig.tight_layout()
        return fig, ax

    # TODO: Add interactive plot
    def plot_predict(self, X, y,
                     storms_to_plot=None,
                     display_info=False,
                     figsize=(15, 10),
                     save=True,
                     file_name='prediction_plot.pdf',
                     plot_persistence=True,
                     model_name=None,
                     sw_to_plot=None,
                     **plot_params):
        check_is_fitted(self)

        storms_to_plot = y.index.unique(level=self.storm_level_)
        if storms_to_plot is not None:
            storms_to_plot = storms_to_plot.intersection(storms_to_plot)

        if len(storms_to_plot) == 0:
            warnings.warn("No storms to plot.")
            # End function
            return None
        elif len(storms_to_plot) > 1:
            idx = pd.IndexSlice
            X = X.loc[idx[storms_to_plot, :], :]
            y = y.loc[idx[storms_to_plot, :]]

        ypred = self.predict(X, y)

        if save:
            pdf = PdfPages(file_name)

        for storm in storms_to_plot:

            # rmse = self.score(X.loc[storm], y.loc[storm])
            fig, ax = self._plot_one_storm(
                X.loc[storm], y.loc[storm],
                storm_idx=storm, display_info=display_info, 
                figsize=figsize, plot_persistence=plot_persistence,
                model_name=model_name, sw_to_plot=sw_to_plot, **plot_params)

            if save:
                pdf.savefig(fig)
            else:
                plt.show()

        if save:
            pdf.close()
        return None

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
        
        
