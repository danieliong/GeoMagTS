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

from GeoMagTS.data_preprocessing import PropagationTimeProcessor, FeatureProcessor, TargetProcessor
from GeoMagTS.utils import _get_NA_mask

# NOTE: Converting to pandas from np slowed down the code significantly


class GeoMagTSRegressor(BaseEstimator, RegressorMixin):
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
                 **estimator_params):
        self.base_estimator = base_estimator.set_params(**estimator_params)
        self.auto_order = auto_order
        self.exog_order = exog_order
        self.pred_step = pred_step
        self.transformer_X = transformer_X
        self.transformer_y = transformer_y
        self.propagate = propagate
        self.time_resolution = time_resolution
        self.D = D
        self.estimator_params = estimator_params

    # def _check_X(self, X, check_multi_index=True, check_vx_col=True, check_same_cols=False):
    #     if not isinstance(X, pd.DataFrame):
    #         raise TypeError("X must be a pandas object (for now).")

    #     if isinstance(X.index, pd.MultiIndex):
    #         if X.index.nlevels < 2:
    #             raise ValueError(
    #                 "X.index must have at least 2 levels corresponding to storm and times if it is a MultiIndex.")
    #     elif check_multi_index:
    #         raise TypeError("X must have a MultiIndex (for now).")
    #     elif not isinstance(X.index, pd.DatetimeIndex):
    #         raise TypeError("Time index must be of type pd.DatetimeIndex.")

    #     # Check if vx_colname is in X
    #     if check_vx_col and self.vx_colname_ not in X.columns:
    #         raise ValueError("X does not have column "+self.vx_colname_+".")

    #     if check_same_cols:
    #         if not X.columns.equals(self.features_):
    #             raise ValueError(
    #                 "X used for predicting must have the same columns as the X used for training.")

    #     dupl_mask_ = X.index.duplicated(keep='last')
    #     if dupl_mask_.any():
    #         warnings.warn(
    #             "Inputs have duplicated indices. Only one of each row with duplicated indices will be kept.")
    #         X = X[~dupl_mask_]

    #     return X, dupl_mask_

    def _check_data(self, X, y=None, fit=True, check_multi_index=True,
                    check_vx_col=True, check_same_cols=False, storm_level=None, time_level=None, **sklearn_check_params):
            
        # Check without converting to np arrays
        _ = check_X_y(X, y, y_numeric=True, **sklearn_check_params)
        
        if fit and y is None:
            return ValueError("y must be specified if fit is True.")
        
        if not fit:
            try:
                check_is_fitted(self)
            except NotFittedError:
                warnings.warn("fit is False and self has not been fitted yet.")
        
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
            except NameError:
                warnings.warn("self.time_level_ is not defined.")
                
                if time_level is not None:
                    warnings.warn("time_level is specified and will be used in removing duplicate times.")
                    dupl_mask_ = X.index.get_level_values(
                        level=time_level).duplicated(keep='last')
                else:
                    raise ValueError("Either self.time_level_ or time_level must be defined if X has a MultiIndex.")
                
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
        
    def _get_target_mask(self, target_processor, propagation_time_processor=None):
        mask = target_processor.mask_

        if self.propagate:
            prop_mask = propagation_time_processor.mask_
            return np.logical_and(mask, prop_mask)
        else:
            return mask
    
    def _process_data(self, X, y=None, fit=True, 
                      check_data=True, remove_NA=True, **check_params):
        
        # Input validation
        if check_data:
            X, y =  self._check_data(X, y, fit=fit, **check_params)
        
        # Process features
        feature_processor_ = FeatureProcessor(
            auto_order=self.auto_order,
            exog_order=self.exog_order,
            time_resolution=self.time_resolution,
            transformer=self.transformer_X)
        if y is not None:
            combined_data = pd.concat([y, X], axis=1)
            X_ = feature_processor_.fit_transform(
                combined_data,
                target_column=0,
                storm_level=self.storm_level_,
                time_level=self.time_level_)
        elif self.auto_order == 0:
            X_ = feature_processor_.fit_transform(
                X, target_column=0, 
                storm_level=self.storm_level_,
                time_level=self.time_level_)
        else:
            raise ValueError("y needs to be given if self.auto_order != 0.")
        
        if fit:
            self.feature_processor_ = feature_processor_
            
            # Process target
            self.target_processor_ = TargetProcessor(
                pred_step=self.pred_step,
                time_resolution=self.time_resolution,
                transformer=self.transformer_y
            )
            y_ = self.target_processor_.fit_transform(
                y, storm_level=self.storm_level_, time_level=self.time_level_)
        
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
        
            target_mask = self._get_target_mask(
            self.target_processor_, self.propagation_time_processor_)
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

        target, features = self._process_data(X, y, fit=True)
        
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
    
    def predict(self, X, y, **predict_params):
        check_is_fitted(self)
        
        X, y = self._check_data(X, y, fit=False, check_multi_index=False, check_vx_col=False, check_same_cols=True)
        
        X_ = self._process_data(X, y, fit=False, 
                                check_data=False,
                                remove_NA=False)
        nan_mask = _get_NA_mask(X_)
        test_features = X_[nan_mask]
        
        # X, _ = self._check_X(X, check_multi_index=False,
        #                      check_vx_col=False, check_same_cols=True)

        # test_feature_processor = FeatureProcessor(
        #     auto_order=self.auto_order,
        #     exog_order=self.exog_order,
        #     time_resolution=self.time_resolution,
        #     transformer=self.transformer_X,
        #     fit_transformer=False)
        # combined_data = pd.concat([y, X], axis=1)
        # X_ = test_feature_processor.fit_transform(
        #     X=combined_data,
        #     target_column=0,
        #     storm_level=self.storm_level_,
        #     time_level=self.time_level_)

        # nan_mask = _get_NA_mask(X_)
        # test_features = X_[nan_mask]

        ypred = self.base_estimator_fitted_.predict(
            test_features, **predict_params)
        
        ypred = self._process_predictions(
            ypred, Vx=X[self.vx_colname_][nan_mask])

        return ypred
    
    def _process_predictions(self, ypred, Vx=None):
        if self.transformer_y is not None:
            check_is_fitted(self.target_processor_.transformer)
            ypred = self.target_processor_.transformer.inverse_transform(
                ypred.reshape(-1, 1)).flatten()
            
        # TODO: Find more efficient way to do this
        if isinstance(Vx.index, pd.MultiIndex):
            times = Vx.index.get_level_values(level=self.time_level_)
            storms = Vx.index.get_level_values(level=self.storm_level_)
            shifted_times = times + pd.Timedelta(minutes=self.pred_step)
        else:
            shifted_times = Vx.index + pd.Timedelta(minutes=self.pred_step)
        
        test_prop_time_processor = PropagationTimeProcessor(
            Vx=Vx.reindex(shifted_times, level=self.time_level_),
            time_resolution=self.time_resolution,
            D=self.D)
        test_prop_time_processor._compute_times(
            storm_level=self.storm_level_,
            time_level=self.time_level_)
        pred_times = test_prop_time_processor.propagated_times_
        
        if isinstance(Vx.index, pd.MultiIndex):
            ypred_processed = pd.Series(ypred, index=[storms, pred_times])
        else:
            ypred_processed = pd.Series(ypred, index=pred_times)
        
        return ypred_processed
    
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
                X, y, persistence=True, squared=False, round=True, decimals=2)
            
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


class GeoMagARX(GeoMagTSRegressor):
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

class GeoMagGP(GeoMagTSRegressor):
    def __init__(self, **params):
        super().__init__(
            base_estimator=GaussianProcessRegressor(),
            **params
            )
        
        
