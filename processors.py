
import torch
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from GeoMagTS.utils import MetaLagFeatureProcessor, _get_NA_mask, mse_masked

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from lazyarray import larray

class GeoMagARXProcessor():
    def __init__(self,
                 auto_order=60,
                 exog_order=60,
                 pred_step=0,
                 transformer_X=None,
                 transformer_y=None,
                 propagate=True,
                 vx_colname='vx_gse',
                 time_resolution='5T',
                 D=1500000,
                 storm_level=0,
                 time_level=1,
                 lazy=True):
        self.auto_order = auto_order
        self.exog_order = exog_order
        self.pred_step = pred_step
        self.transformer_X = transformer_X
        self.transformer_y = transformer_y
        self.propagate = propagate
        self.vx_colname = vx_colname
        self.time_resolution = time_resolution
        self.D = D
        self.storm_level = storm_level
        self.time_level = time_level
        self.lazy = lazy
        self.processor_fitted_ = False

    def check_data(self, X, y=None, fit=True, check_multi_index=True,
                   check_vx_col=True, check_same_cols=False, check_same_indices=True, **sklearn_check_params):

        # if not remove_duplicates:
        #     # Don't need to copy since X, y are not manipulated.
        #     copy=False

        # Check without converting to np arrays
        _ = check_X_y(X, y, y_numeric=True, **sklearn_check_params)

        if fit and y is None:
            return ValueError("y must be specified if fit is True.")

        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise TypeError("X must be a pandas object (for now).")

        if isinstance(X.index, pd.MultiIndex):
            if X.index.nlevels < 2:
                raise ValueError(
                    "X.index must have at least 2 levels corresponding to storm and times if it is a MultiIndex.")
                
            # if remove_duplicates:
            #     # Get mask for removing duplicates
            #     dupl_mask_ = X.index.get_level_values(
            #         level=self.time_level).duplicated(keep='last')

        elif check_multi_index:
            raise TypeError("X must have a MultiIndex (for now).")

        elif not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("Time index must be of type pd.DatetimeIndex.")

        # if fit and remove_duplicates:
        #     # Save dupl_mask_
        #     self.dupl_mask_ = dupl_mask_

        # if remove_duplicates:
        #     if dupl_mask_.any():
        #         warnings.warn(
        #             "Inputs have duplicated indices. Only one of each row with duplicated indices will be kept.")

        # Check if vx_colname is in X
        if check_vx_col and self.vx_colname not in X.columns:
            raise ValueError("X does not have column "+self.vx_colname_+".")

        if check_same_cols:
            if not X.columns.equals(self.features_):
                raise ValueError(
                    "X must have the same columns as the X used for fitting.")
        
        if y is not None:
            if not isinstance(y, (pd.Series, pd.DataFrame)):
                raise TypeError("y must be a pandas object (for now).")

            # Check if X, y have same times
            if check_same_indices and not X.index.equals(y.index):
                raise ValueError("X, y do not have the same indices.")
            
            # if remove_duplicates:
            #     y_ = y_[~dupl_mask_]

        # if remove_duplicates:
        #     X_ = X_[~dupl_mask_]

        # return X_, y_
        return None
    
    def _remove_duplicates(self, X, y=None, fit=True):
        
        if isinstance(X.index, pd.MultiIndex):
            dupl_mask_ = X.index.get_level_values(
                level=self.time_level).duplicated(keep='last')
        else:
            dupl_mask_ = X.index.duplicated(keep='last')
            
        if dupl_mask_.any():
            warnings.warn(
                "Inputs have duplicated indices. Only one of each row with duplicated indices will be kept.")
        
        if fit:
            self.dupl_mask_ = larray(dupl_mask_)
        
        if y is not None:
            if not X.index.equals(y.index):
                raise ValueError("X, y do not have the same indices.")
            
        # This will (probably) remove duplicates in original X, y
            return X[~dupl_mask_], y[~dupl_mask_]
        else:
            return X[~dupl_mask_]
        
    def process_data(self, X, y=None,  
                     fit=True, 
                     check_data=True,
                     remove_NA=True,
                     remove_duplicates=True,
                     copy=True, **check_params):

        if fit:
            self.features_ = X.columns
        elif not self.processor_fitted_:
            raise NotProcessedError(self)
        
        X_ = X
        y_ = y
        if copy:
            X_ = X_.copy()
            if y_ is not None:
                y_ = y_.copy()

        if check_data:
            self.check_data(X_, y_, fit=fit, check_vx_col=self.propagate, 
                check_same_cols=(not fit), **check_params)
        # elif copy and y_ is not None:
        #     X_ = X.copy()
        #     y_ = y.copy()
        # elif copy and y_ is None:
        #     X_ = X.copy()
        
        if remove_duplicates:
            X_, y_ = self._remove_duplicates(X_, y_, fit=fit)
            
        if y_ is not None:
            data_ = pd.concat([y_, X_], axis=1)
        elif self.auto_order == 0:
            data_ = X_
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
                storm_level=self.storm_level,
                time_level=self.time_level)

            # Fit target processor and transform training targets
            self.target_processor_ = TargetProcessor(
                pred_step=self.pred_step,
                time_resolution=self.time_resolution,
                transformer=self.transformer_y
            )
            y_ = self.target_processor_.fit_transform(
                y_, storm_level=self.storm_level, time_level=self.time_level)

            # Fit propagation time processor and transform training targets
            if self.propagate:
                self.propagation_time_processor_ = PropagationTimeProcessor(
                    Vx=X[self.vx_colname],
                    time_resolution=self.time_resolution,
                    D=self.D)
                y_ = self.propagation_time_processor_.fit_transform(
                    y_, time_level=self.time_level)
            else:
                self.propagation_time_processor_ = None

            # BUG: lengths of X_, y_ differ when pred_step > 0

            self.processor_fitted_ = True
        elif y is not None and self.transformer_y is not None:
            # Don't transform y_ 
            y_ = self.transformer_y.transform(
                y_.to_numpy().reshape(-1,1)).flatten()

        # Use feature_processor that was fit previously.
        X_ = self.feature_processor_.transform(data_)

        if fit:
            # Align X_ with y_
            target_mask = self.target_processor_.mask_
            if self.propagate:
                target_mask = np.logical_and(
                    target_mask, self.propagation_time_processor_.mask_)
            X_ = X_[target_mask]
        
        if remove_NA:
            mask = _get_NA_mask(X_, y_, mask_y=fit)
            X_ = X_[mask]
            
            if y is not None:
                y_ = y_[mask]
        
        if fit:
            self.train_features_ =  X_
            self.train_target_ = y_
        
        return X_, y_
    
    def _check_processor_fitted(self, check_lazy=True):
        if not self.processor_fitted_:
            raise NotProcessedError(self)
        if check_lazy and not self.lazy:
            raise ValueError("self.lazy must be True.")
    
    @property
    def train_features_(self):
        if self.lazy:
            return self.train_features__.evaluate()
        else:
            return self.train_features__
    
    @train_features_.setter
    def train_features_(self, train_features_):
        if self.lazy:
            self.train_features__ = larray(train_features_)
        else:
            self.train_features__ = train_features_
            
    @property
    def train_target_(self):
        if self.lazy:
            return self.train_target__.evaluate()
        else:
            return self.train_target__
    
    @train_target_.setter
    def train_target_(self, train_target_):
        if self.lazy:
            self.train_target__ = larray(train_target_)
        else:
            self.train_target__ = train_target_
            
    @property
    def train_shape_(self):
        self._check_processor_fitted(check_lazy=False)
        return self.train_features__.shape
        
    
    def process_predictions(self, ypred, Vx=None, 
                             inverse_transform_y=True, copy=True):
        ypred_ = ypred
        if copy:
            if isinstance(ypred_, (np.ndarray, pd.Series, pd.DataFrame)):
                ypred_ = ypred_.copy()
            elif isinstance(ypred_, torch.Tensor):
                ypred_ = ypred_.clone()
        
        if self.transformer_y is not None and inverse_transform_y:
            check_is_fitted(self.target_processor_.transformer)
            ypred_ = self.target_processor_.transformer.inverse_transform(
                ypred_.reshape(-1, 1)).flatten()
            ypred_ = pd.Series(ypred_, index = ypred.index)
        # elif isinstance(ypred_, torch.Tensor):
        #     ypred_ = ypred_.numpy()
        # else:
        #     if isinstance(ypred_, (pd.Series, pd.DataFrame)):
        #         ypred_ = ypred_.to_numpy()
        #     elif isinstance(ypred_, torch.Tensor):
        #         ypred_ = ypred_.numpy()

        # TODO: Find more efficient way to do this
        # if isinstance(Vx.index, pd.MultiIndex):
        #     times = Vx.index.get_level_values(level=self.time_level)
        #     # storms = Vx.index.get_level_values(level=self.storm_level)
        #     shifted_times = times + pd.Timedelta(minutes=self.pred_step)
        # else:
        #     shifted_times = Vx.index + pd.Timedelta(minutes=self.pred_step)

        Vx_ = Vx
        if copy:
            if isinstance(Vx_, (np.ndarray, pd.Series, pd.DataFrame)):
                Vx_ = Vx_.copy()
            elif isinstance(Vx_, torch.Tensor):
                Vx_ = Vx_.clone()
        
        test_target_processor = TargetProcessor(
            pred_step=self.pred_step,
            time_resolution=self.time_resolution)
        Vx_ = test_target_processor.fit_transform(
            Vx_, storm_level=self.storm_level, time_level=self.time_level)
            
        test_prop_time_processor = PropagationTimeProcessor(
            Vx=Vx_,
            time_resolution=self.time_resolution,
            D=self.D)
        test_prop_time_processor._compute_times(
            storm_level=self.storm_level,
            time_level=self.time_level)
        mask = test_prop_time_processor._compute_mask(Vx_)
        pred_times = test_prop_time_processor.propagated_times[mask]

        if isinstance(Vx.index, pd.MultiIndex):
            ypred_ = pd.Series(ypred_[mask], 
                               index=[pred_times.index, pred_times])
        else:
            ypred_ = pd.Series(ypred_[mask], index=pred_times)
        
        # Remove duplicate indices that may have resulted from 
        # processing propagation time
        ypred_ = self._remove_duplicates(ypred_, fit=False)
        # BUG: This breaks down if we have more than one testing storm. Fix later.
        
        return ypred_
    
    def _predict_persistence(self, y, Vx=None):
        if isinstance(y.index, pd.MultiIndex):
            ypred = y.groupby(level=self.storm_level_).apply(
                lambda x: x.unstack(level=self.storm_level_).shift(
                    periods=self.pred_step, freq='T'
                )
            )
        else:
            ypred = y.shift(periods=self.pred_step, freq='T')
        
        if self.propagate:
            pers_prop_time_processor = PropagationTimeProcessor(
                Vx=Vx.reindex(ypred.index).dropna(), 
                time_resolution=self.time_resolution,
                D=self.D)
            pers_prop_time_processor = pers_prop_time_processor.fit(
                ypred, storm_level=self.storm_level, 
                time_level=self.time_level)
            mask = pers_prop_time_processor.mask_
            prop_times = pers_prop_time_processor.propagated_times[mask] 
            
            if isinstance(y.index, pd.MultiIndex):
                ypred = pd.Series(
                    ypred.values.flatten()[mask],
                    index=[prop_times.index, prop_times],
                    )
            else:
                ypred = pd.Series(
                    ypred.values[mask], 
                    index=prop_times.values)
            
            ypred = self._remove_duplicates(ypred, fit=False)
            
        return ypred
    
    def _plot_one_storm(self, storm_idx, y, ypred, X=None,
                        ypred_persistence=None, lower=None, upper=None, display_info=False, figsize=(10, 7), 
                        model_name=None, sw_to_plot=None, **more_info):

        if sw_to_plot is not None:
            if X is None:
                raise ValueError("X needs to be specified if sw_to_plot is specified.")
            
            n_sw_to_plot = len(sw_to_plot)
            fig, ax = plt.subplots(nrows=n_sw_to_plot+1, 
                                   ncols=1,
                                   sharex=True,
                                   figsize=figsize,
                                   gridspec_kw={
                                       'height_ratios':[4]+[1]*n_sw_to_plot})
            ax0 = ax[0]
        else:
            fig, ax = plt.subplots(sharex=True,
                                   figsize=figsize)
            ax0 = ax
        
        ### Plot truth
        # y_.unstack(level=self.storm_level).plot(label='Truth', color='black', linewidth=0.5, ax=ax0)
        ax0.plot(y[storm_idx], 
                 label='Truth', color='black', linewidth=0.5)
        
        ### Plot predictions
        rmse = mse_masked(y, ypred, squared=False, round=True)
    
        # Get prediction label
        pred_label = ''
        if self.propagate:
            pred_label = pred_label + 'Propagated '
        if self.pred_step > 0:
            pred_label = pred_label + str(self.pred_step) + 'min. ahead '
        pred_label = pred_label + 'prediction (RMSE: '+ str(rmse)+')'
        
        # ypred_.unstack(level=self.storm_level).plot(
        #     label=pred_label, color='red', linewidth=0.5, ax=ax0)        
        ax0.plot(ypred[storm_idx], 
                 label=pred_label, color='red', linewidth=0.5)

        if lower is not None and upper is not None:
            ax0.fill_between(
                lower.index.get_level_values(level=self.time_level),
                            lower[storm_idx], upper[storm_idx], 
                            alpha=0.25, color='red')

        if ypred_persistence is not None:
                rmse_persistence = mse_masked(y, ypred_persistence, 
                                            squared=False, round=True)
                
                persistence_label = 'Persistence (RMSE: '+str(rmse_persistence)+')'
                
                # ypred_persistence.unstack(level=self.storm_level).plot(
                #     label=persistence_label, color='blue', linestyle='--',
                #     linewidth=0.5)
                ax0.plot(ypred_persistence[storm_idx],
                         label=persistence_label,color='blue', 
                         linestyle='--', linewidth=0.5)

        ax0.legend()
        # Adjust time scale
        locator = mdates.AutoDateLocator(minticks=15)
        formatter = mdates.ConciseDateFormatter(locator)
        ax0.xaxis.set_major_locator(locator)
        ax0.xaxis.set_major_formatter(formatter)

        if display_info:
            info = 'Storm #'+str(storm_idx)+": " + \
            'auto_order='+str(self.auto_order)+', ' + \
            'exog_order='+str(self.exog_order)+' (in min.) '
            
            if model_name is not None:
                info = info + '[Model: ' + model_name + ']'
            
            if len(more_info) != 0:
                info = info + ' ('
                i = 0
                for param, value in more_info.items():
                    info = info + param + '=' + str(value)
                    if i != len(more_info)-1:
                        info = info + ', '
                        i = i + 1
                info = info + ')'
            ax0.set_title(info)
        
        if sw_to_plot is not None:
            
            for i in range(n_sw_to_plot):
                ax[i+1].plot(X[sw_to_plot[i]][storm_idx], label=sw_to_plot[i], 
                        color='black', linewidth=0.5)
                ax[i+1].legend()
                ax[i+1].xaxis.set_major_locator(locator)
                ax[i+1].xaxis.set_major_formatter(formatter)
                       
            fig.tight_layout()
        return fig, ax

    def plot_predict(self, y, ypred, X=None,
                     upper=None, lower=None,
                     plot_persistence=True,
                     storms_to_plot=None,
                     display_info=False,
                     figsize=(15, 10),
                     save=True,
                     file_name='prediction_plot.pdf',
                     model_name=None,
                     sw_to_plot=None,
                     **more_info):
        # TODO: Need to handle case where y, ypred don't have MultiIndex

        storms_to_plot = y.index.unique(level=self.storm_level)
        if storms_to_plot is not None:
            storms_to_plot = storms_to_plot.intersection(storms_to_plot)

        if len(storms_to_plot) == 0:
            warnings.warn("No storms to plot.")
            # End function
            return None
        elif len(storms_to_plot) > 1:
            idx = pd.IndexSlice
            ypred = ypred.loc[idx[storms_to_plot, :]]
            y = y.loc[idx[storms_to_plot, :]]
            
        if plot_persistence:
            if X is None:
                raise ValueError("X needs to specified if plot_persistence is True.")
            
            ypred_persistence = self._predict_persistence(
                y, Vx=X[self.vx_colname])
        else:
            ypred_persistence = None
            
        if save:
            pdf = PdfPages(file_name)

        for storm in storms_to_plot:

            # rmse = self.score(X.loc[storm], y.loc[storm])
            fig, ax = self._plot_one_storm(storm,
                y, ypred, X=X, ypred_persistence=ypred_persistence, 
                lower=lower, upper=upper, display_info=display_info,
                figsize=figsize, model_name=model_name, 
                sw_to_plot=sw_to_plot, **more_info)

            if save:
                pdf.savefig(fig)
            else:
                plt.show()

        if save:
            pdf.close()
        return None

    
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
            self.transformer = self.transformer.set_params(
                **self.transformer_params)

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
                 D=1500000,
                 lazy=True):
        self.Vx = Vx
        self.time_resolution = time_resolution
        self.D = D
        self.lazy = lazy

    def _compute_propagated_time(self, x):
        return (x['time'] + pd.Timedelta(x['prop_time'], unit='sec')).floor(freq=self.time_resolution)

    def _compute_times(self, storm_level=0, time_level=1):
        propagation_in_sec_ = self.D / self.Vx.abs()
        propagation_in_sec_.rename("prop_time", inplace=True)

        if isinstance(propagation_in_sec_.index, pd.MultiIndex):
            propagation_in_sec_.index.rename(
                names='time', level=time_level, inplace=True)
            propagated_times_ = propagation_in_sec_.reset_index(
                level=time_level).apply(self._compute_propagated_time, axis=1)
        else:
            propagation_in_sec_.index.rename('time', inplace=True)
            propagated_times_ = propagation_in_sec_.reset_index().apply(
                self._compute_propagated_time, axis=1)
            propagated_times_.rename('times', inplace=True)
        
        self.propagation_in_sec = propagation_in_sec_
        self.propagated_times = propagated_times_

        return None
    
    def _compute_mask(self, X, time_level=1):
        # Get mask of where propagated times are in X's time index
        if isinstance(X.index, pd.MultiIndex):
            X_times = X.index.get_level_values(level=time_level)
            proptime_in_X_mask = np.in1d(self.propagated_times, X_times)
        else:
            proptime_in_X_mask = np.in1d(self.propagated_times, X.index)

        # Get mask of where propagated times are duplicated
        dupl_mask = self.propagated_times.duplicated(keep='last').values
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
        self.mask_ = self._compute_mask(X, time_level=self.time_level_)

        return self
        
    def transform(self, X, y=None):
        check_is_fitted(self)

        if self.Vx is not None:
            if isinstance(X.index, pd.MultiIndex):
                storms = X.index.get_level_values(level=self.storm_level_)
                return X.reindex(
                    [storms[self.mask_], self.propagated_times[self.mask_]]
                )
            else:
                return X.reindex(
                    [self.mask_])
        else:
            return X
    
    @property
    def propagation_in_sec(self):
        if self.lazy:
            # Reconstruct series
            df = pd.Series(self.propagation_in_sec_.evaluate())
            if len(self.propagation_in_sec_idx_names_) > 1:
                df.index = pd.MultiIndex.from_tuples(
                    self.propagation_in_sec_idx_.evaluate(),
                    names=self.propagation_in_sec_idx_names_)
            else:
                df.set_index(self.propagation_in_sec_idx_, inplace=True)

            return df
        else:
            return self.propagation_in_sec_

    @propagation_in_sec.setter
    def propagation_in_sec(self, propagation_in_sec):        
        if self.lazy:
            if isinstance(propagation_in_sec, (pd.DataFrame, pd.Series)):
                # Save index and index names to recreate pd object later.
                self.propagation_in_sec_idx_names_ = propagation_in_sec.index.names
                self.propagation_in_sec_idx_ = larray(propagation_in_sec.index)
                
            self.propagation_in_sec_ = larray(propagation_in_sec)
        else:
            self.propagation_in_sec_ = propagation_in_sec
            
    @property
    def propagated_times(self):
        if self.lazy:
            # Reconstruct series 
            df = pd.Series(self.propagated_times_.evaluate())
            if len(self.propagated_times_idx_names_) > 1:
                df.index = pd.MultiIndex.from_tuples(
                    self.propagated_times_idx_.evaluate(),
                    names=self.propagated_times_idx_names_)
            else:
                df.index = self.propagated_times_idx_
            return df
        else:
            return self.propagated_times_

    @propagated_times.setter
    def propagated_times(self, propagated_times):
        if self.lazy:
            if isinstance(propagated_times, (pd.DataFrame, pd.Series)):
                # Save index and index names to recreate pd object later.
                self.propagated_times_idx_names_ = propagated_times.index.names
                self.propagated_times_idx_ = larray(propagated_times.index)

            self.propagated_times_ = larray(propagated_times)
        else:
            self.propagated_times_ = propagated_times

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
        X_ = X_.reindex(
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
