import warnings
from functools import wraps

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from pandas.tseries.frequencies import to_offset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..utils import (MetaLagFeatureProcessor, _check_file_name, _get_NA_mask,
                     _pd_fit, _pd_transform, mse_masked)

# NOTE: Get rid of sklearn-transformers?


def requires_processor_fitted(method):
    @wraps(method)
    def wrapped_method(self, *args, **kwargs):
        if not self.processor_fitted_:
            raise NotProcessedError(self)
        return method(self, *args, **kwargs)

    return wrapped_method


class NotProcessedError(Exception):
    def __init__(self, obj):
        # class_name = obj.__class__.__name__
        self.message = (
            "Data have not been previously processed using process_data in " +
            obj.__class__.__name__ + ". Please call" + obj.__class__.__name__ +
            ".process_data with fit=True first.")


class GeoMagARXProcessor():
    def __init__(self,
                 auto_order=60,
                 exog_order=60,
                 pred_step=0,
                 transformer_X=None,
                 transformer_y=None,
                 include_interactions=False,
                 interactions_degree=2,
                 seasonality=False,
                 propagate=True,
                 vx_colname='vx_gse',
                 time_resolution='5T',
                 D=1500000,
                 storm_level=0,
                 time_level=1,
                 save_train=False,
                 lazy=True):
        # TODO: Remove lazy. It literally does nothing.
        # TODO: Add save_data arg.
        """Process geomagnetic data for fitting autoregressive models

        Parameters
        ----------
        auto_order : int, optional
            Autoregressive order in minutes, by default 60
        exog_order : int, optional
            Exogeneous order in minutes, by default 60
        pred_step : int, optional
            Prediction step in minutes, by default 0
        transformer_X : sklearn.Transformer, optional
            Scikit-learn transformer for feature variables, by default None
        transformer_y : sklearn.Transformer, optional
            Scikit-learn transformer for target variable, by default None
        include_interactions : bool, optional
            If true, include interaction terms, by default False
        interactions_degree : int, optional
            If include_interactions is true, the degree of interactions, by default 2
        seasonality : bool, optional
            If true, include terms to account for day and year seasonality, by default False
        propagate : bool, optional
            If true, propagate time for target variable, by default True    
        vx_colname : str, optional
            Name of column for Vx, solar wind velocity, by default 'vx_gse'
        time_resolution : str, datetime.timedelta, or pandas.DateOffset, optional
            Time resolution, by default '5T'
        D : int, optional
            Distance from Earth to spacecraft that measures solar wind, by default 1500000
        storm_level : int, optional
            Index level of storms in data, by default 0
        time_level : int, optional
            Index level of time in data, by default 1
        lazy : bool, optional
            If true, save lazy evaluated arrays, by default True
        """

        self.auto_order = auto_order
        self.exog_order = exog_order
        self.pred_step = pred_step
        self.transformer_X = transformer_X
        self.transformer_y = transformer_y
        self.include_interactions = include_interactions
        self.interactions_degree = interactions_degree
        self.seasonality = seasonality
        self.propagate = propagate
        self.vx_colname = vx_colname

        # FIXME: This shouldn't be input.
        self.time_resolution = time_resolution

        self.D = D
        self.storm_level = storm_level
        self.time_level = time_level
        self.lazy = lazy
        self.save_train = save_train
        self.processor_fitted_ = False

    @property
    def time_res_minutes_(self):
        """
        Time resolution in minutes
        """
        return to_offset(self.time_resolution).delta.seconds / 60

    @property
    def auto_order_steps_(self):
        """
        Autoregressive order time-step
        """
        return np.rint(self.auto_order / self.time_res_minutes_).astype(int)

    @property
    def exog_order_steps_(self):
        """
        Autoregressive order time-step
        """
        return np.rint(self.exog_order / self.time_res_minutes_).astype(int)

    @property
    @requires_processor_fitted
    def interactions_processor_(self):
        """Scikit-learn PolynomialFeatures transformer for creating interaction terms 

        Returns
        -------
        sklearn.Transformer
            PolynomialFeatures transformer 
        """
        if self.include_interactions:
            if self.transformer_X is not None:
                return self.feature_processor_.transformer[
                    'polynomialfeatures']
            else:
                return self.feature_processor_.transformer
        else:
            return None

    # @property
    # @requires_processor_fitted
    # def train_features_(self):
    #     """
    #     Features data used for training
    #     """
    #     if self.lazy:
    #         return self.train_features__.evaluate()
    #     else:
    #         return self.train_features__

    # @##train_features_.setter
    # def train_features_(self, train_features_):
    #     if self.lazy:
    #         self.train_features__ = larray(train_features_)
    #     else:
    #         self.train_features__ = train_features_

    # @property
    # @requires_processor_fitted
    # def train_target_(self):
    #     """
    #     Target data used for training
    #     """
    #     if self.lazy:
    #         return self.train_target__.evaluate()
    #     else:
    #         return self.train_target__

    # @train_target_.setter
    # def train_target_(self, train_target_):
    #     if self.lazy:
    #         self.train_target__ = larray(train_target_)
    #     else:
    #         self.train_target__ = train_target_

    # @property
    # @requires_processor_fitted
    # def train_storms_(self):
    #     """
    #     Storms used for training
    #     """
    #     if self.lazy:
    #         return self.train_storms__.evaluate()
    #     else:
    #         return self.train_storms__

    # @train_storms_.setter
    # def train_storms_(self, train_storms_):
    #     if self.lazy:
    #         self.train_storms__ = larray(train_storms_)
    #     else:
    #         self.train_storms__ = train_storms_

    # @property
    # @requires_processor_fitted
    # def train_shape_(self):
    #     return self.train_features__.shape

    @property
    def dupl_mask_(self):
        if self.lazy:
            return self.dupl_mask__.evaluate()
        else:
            return self.dupl_mask__

    @dupl_mask_.setter
    def dupl_mask_(self, dupl_mask_):
        if self.lazy:
            self.dupl_mask__ = larray(dupl_mask_)
        else:
            self.dupl_mask__ = dupl_mask_

    def check_data(self,
                   X,
                   y=None,
                   fit=True,
                   check_multi_index=True,
                   check_vx_col=True,
                   check_same_cols=False,
                   check_same_indices=True,
                   **sklearn_check_params):
        """Input validation for data

        Parameters
        ----------
        X : pd.DataFrame
            Features data
        y : pd.Series, optional
            Target data, by default None
        fit : bool, optional
            If True, check data used for fitting, by default True
        check_multi_index : bool, optional
            If True, check if index is a MultiIndex, by default True
        check_vx_col : bool, optional
            If True, check if X contains a column named self.vx_colname, by default True
        check_same_cols : bool, optional
            If True, check if X has columns self.train_features_cols_, by default False
        check_same_indices : bool, optional
            If True, check if X and y have same indices, by default True

        Returns
        -------
        None

        Raises
        ------            
        ValueError
            y is not specified (if fit is True), X has a MultiIndex with only
            one level, X does not contain column named self.vx_colname (if
            check_vx_col is True), X does not have columns
            self.train_features_cols_ (if check_same_cols is True), or X does not
            have same indices as y (if check_same_indices is True)
        TypeError
            X is not a pandas object, X does not have a MultiIndex (if
            check_multi_index) or DatetimeIndex, or y is not a pandas object (if y
            is not None)
        """
        # Check without converting to np arrays
        _ = check_X_y(X, y, y_numeric=True, **sklearn_check_params)

        if fit and y is None:
            return ValueError("y must be specified if fit is True.")

        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise TypeError("X must be a pandas object (for now).")

        if isinstance(X.index, pd.MultiIndex):
            if X.index.nlevels < 2:
                raise ValueError(
                    "X.index must have at least 2 levels corresponding to storm and times if it is a MultiIndex."
                )

        elif check_multi_index:
            raise TypeError("X must have a MultiIndex (for now).")

        elif not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("Time index must be of type pd.DatetimeIndex.")

        # Check if vx_colname is in X
        if check_vx_col and self.vx_colname not in X.columns:
            raise ValueError("X does not have column " + self.vx_colname
                             + ".")

        if check_same_cols:
            if not X.columns.equals(self.train_features_cols_):
                raise ValueError(
                    "X must have the same columns as the X used for fitting.")

        if y is not None:
            if not isinstance(y, (pd.Series, pd.DataFrame)):
                raise TypeError("y must be a pandas object (for now).")

            # Check if X, y have same times
            if check_same_indices and not X.index.equals(y.index):
                raise ValueError("X, y do not have the same indices.")

        return None

    def _compute_duplicate_time_mask(self, X):
        """Compute mask for duplicated time in X 

        Parameters
        ----------
        X : pd.DataFrame
            Feature data_

        Returns
        -------
        np.ndarray of boolean values
            Mask for times that aren't duplicates
        """
        if isinstance(X.index, pd.MultiIndex):
            dupl_mask_ = ~X.index.get_level_values(
                level=self.time_level).duplicated(keep='last')
        else:
            dupl_mask_ = ~X.index.duplicated(keep='last')

        if not dupl_mask_.all():
            warnings.warn(
                "Inputs have duplicated indices. Only one of each row with duplicated indices will be kept."
            )

        return dupl_mask_

    def process_data(self,
                     X,
                     y=None,
                     fit=True,
                     check_data=True,
                     remove_NA=True,
                     remove_duplicates=True,
                     copy=True,
                     **check_params):
        """Process data for fitting AR-X models to geomagnetic data

        Parameters
        ----------
        X : pd.DataFrame
            Feature data
        y : pd.Series, optional
            Target data, by default None
        fit : bool, optional
            Process data for model fitting if true and for prediction if false, by default True
        check_data : bool, optional
            If True, perform input validation for data, by default True
        remove_NA : bool, optional
            If True, remove NA values, by default True
        remove_duplicates : bool, optional
            If True, remove duplicate times, by default True
        copy : bool, optional
            If True, create deep copies of data, by default True

        Returns
        -------
        tuple of np.ndarray
            Tuple of numpy arrays with columns containing autoregressive and
            lagged exogeneous terms to be used as features and target in
            regression model

        Raises
        ------
        NotProcessedError
            If fit is False (process data for prediction) and self.processor_fitted_ is False
        ValueError
            y is not specified and self.auto_order is non-zero.
        """

        if fit:
            self.train_features_cols_ = X.columns
        elif not self.processor_fitted_:
            raise NotProcessedError(self)

        X_ = X
        y_ = y
        if copy:
            X_ = X_.copy()
            if y_ is not None:
                y_ = y_.copy()

        if check_data:
            self.check_data(X_,
                            y_,
                            fit=fit,
                            check_vx_col=self.propagate,
                            check_same_cols=(not fit),
                            **check_params)

        if remove_duplicates:
            self.dupl_mask_ = self._compute_duplicate_time_mask(X_)
            X_ = X_[self.dupl_mask_]
            if y is not None:
                y_ = y_[self.dupl_mask_]
        else:
            self.dupl_mask_ = np.array([True] * X_.shape[0])

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
            self.feature_processor_ = ARXFeatureProcessor(
                auto_order=self.auto_order,
                exog_order=self.exog_order,
                include_interactions=self.include_interactions,
                interactions_degree=self.interactions_degree,
                seasonality=self.seasonality,
                time_resolution=self.time_resolution,
                transformer=self.transformer_X)
            self.feature_processor_.fit(data_,
                                        target_column=0,
                                        storm_level=self.storm_level,
                                        time_level=self.time_level)

            # Fit target processor and transform training targets
            self.target_processor_ = TargetProcessor(
                pred_step=self.pred_step,
                time_resolution=self.time_resolution,
                transformer=self.transformer_y)
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

            # BUG: lengths of X_, y_ may differ when pred_step > 0

            self.processor_fitted_ = True
        elif y is not None and self.transformer_y is not None:
            # Don't transform y_
            y_ = self.transformer_y.transform(y_.to_numpy().reshape(
                -1, 1)).flatten()

        # Use feature_processor that was fit previously.
        X_ = self.feature_processor_.transform(data_)
        # Doesn't change shape of X_

        if fit:
            # Align X_ with y_
            target_mask = self.target_processor_.mask_
            if self.propagate:
                target_mask = np.logical_and(
                    target_mask, self.propagation_time_processor_.mask_)
            X_ = X_[target_mask]

        if remove_NA:
            na_mask = _get_NA_mask(X_, y_, mask_y=fit)
            X_ = X_[na_mask]

            if y is not None:
                y_ = y_[na_mask]

        # Get storms of training data
        if fit and isinstance(X.index, pd.MultiIndex):
            input_storms = X.index.get_level_values(level=self.storm_level)
            if self.save_train:
                self.train_storms_ = input_storms[
                    self.dupl_mask_][target_mask][na_mask]

        y_ = np.array(y_)

        if fit and self.save_train:
            self.train_features_ = X_
            self.train_target_ = y_

        return X_, y_

    def process_predictions(self,
                            ypred,
                            Vx=None,
                            inverse_transform_y=True,
                            copy=True):
        """Process predictions: perform inverse transform if y was transformed,compute corresponding times for predictions 

        Parameters
        ----------
        ypred : np.ndarray
            Predictions
        Vx : array-like, optional
            Solar wind speed used for computing propagation time, by default None
        inverse_transform_y : bool, optional
            if True, perform inverse transform, by default True
        copy : bool, optional
            if True, create deep copies, by default True

        Returns
        -------
        pd.Series
            Processed predictions
        """
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
            # ypred_ = pd.Series(ypred_, index=ypred.index)
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
            pred_step=self.pred_step, time_resolution=self.time_resolution)
        Vx_ = test_target_processor.fit_transform(Vx_,
                                                  storm_level=self.storm_level,
                                                  time_level=self.time_level)

        test_prop_time_processor = PropagationTimeProcessor(
            Vx=Vx_, time_resolution=self.time_resolution, D=self.D)
        test_prop_time_processor._compute_times(storm_level=self.storm_level,
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
        dupl_mask = self._compute_duplicate_time_mask(ypred_)
        ypred_ = ypred_[dupl_mask]
        # BUG: This breaks down if we have more than one testing storm. Fix later.

        return ypred_

    def _predict_persistence(self, y, Vx=None):
        if isinstance(y.index, pd.MultiIndex):
            ypred = y.groupby(level=self.storm_level).apply(
                lambda x: x.unstack(level=self.storm_level).shift(
                    periods=self.pred_step, freq='T'))
        else:
            ypred = y.shift(periods=self.pred_step, freq='T')

        if self.propagate:
            pers_prop_time_processor = PropagationTimeProcessor(
                Vx=Vx.reindex(ypred.index).dropna(),
                time_resolution=self.time_resolution,
                D=self.D)
            pers_prop_time_processor = pers_prop_time_processor.fit(
                ypred,
                storm_level=self.storm_level,
                time_level=self.time_level)
            mask = pers_prop_time_processor.mask_
            prop_times = pers_prop_time_processor.propagated_times[mask]

            if isinstance(y.index, pd.MultiIndex):
                # BUG: Breaks down when there are > 1 test storms.
                ypred = pd.Series(
                    ypred.values.flatten()[mask],
                    index=[prop_times.index, prop_times],
                )
            else:
                ypred = pd.Series(ypred.values[mask], index=prop_times.values)

            dupl_mask = self._compute_duplicate_time_mask(ypred)
            ypred = ypred[dupl_mask]

        return ypred

    def _plot_one_storm(self,
                        storm_idx,
                        y,
                        ypred,
                        X=None,
                        ypred_persistence=None,
                        lower=None,
                        upper=None,
                        display_info=False,
                        figsize=(10, 7),
                        model_name=None,
                        sw_to_plot=None,
                        time_range=None,
                        min_ticks=10,
                        interactive=False,
                        y_orig=None,
                        **more_info):

        y_ = y[storm_idx]
        ypred_ = ypred[storm_idx]
        X_ = X.loc[storm_idx]
        ypred_persistence_ = ypred_persistence[storm_idx]

        # Get prediction label
        rmse = mse_masked(y, ypred, squared=False, round=True)
        pred_label = 'Prediction'
        if self.propagate or self.pred_step > 0:
            pred_label = pred_label + ' ('
        if self.propagate:
            pred_label = pred_label + 'propagated'
        if self.pred_step > 0:
            if self.propagate:
                pred_label = pred_label + ', '

            pred_label = pred_label + str(self.pred_step) + 'min. ahead'
        if self.propagate or self.pred_step > 0:
            pred_label = pred_label + ')'
        pred_label = pred_label + ' [RMSE: ' + str(rmse) + ']'

        info = 'Storm #'+str(storm_idx)+": " + \
            'auto_order='+str(self.auto_order)+'min, ' + \
            'exog_order='+str(self.exog_order)+'min'

        if self.include_interactions:
            info = info + ', ' + \
                str(self.interactions_degree) + '-way interactions'

        if model_name is not None:
            info = info + ' [Model: ' + model_name + ']'

        if len(more_info) != 0:
            info = info + ' ('
            i = 0
            for param, value in more_info.items():
                info = info + param + '=' + str(value)
                if i != len(more_info) - 1:
                    info = info + ', '
                    i = i + 1
            info = info + ')'

        if ypred_persistence is not None:
            rmse_persistence = mse_masked(y_,
                                          ypred_persistence_,
                                          squared=False,
                                          round=True)

            persistence_label = 'Persistence [RMSE: ' + \
                str(rmse_persistence)+']'

        if sw_to_plot is not None:
            if X is None:
                raise ValueError(
                    "X needs to be specified if sw_to_plot is specified.")

            n_sw_to_plot = len(sw_to_plot)

        features_text = 'Features: ' + ', '.join(self.train_features_cols_)

        if interactive:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            if sw_to_plot is not None:
                fig = make_subplots(rows=len(sw_to_plot) + 1,
                                    cols=1,
                                    row_heights=[4] + [1] * n_sw_to_plot,
                                    shared_xaxes=True,
                                    vertical_spacing=0.02)
            else:
                fig = go.Figure()

            # hovertemplate =  'Time: %{x}' + '<br>value: %{y:.2f}'

            # Plot truth
            fig.add_trace(
                go.Scatter(
                    x=y_.index,
                    y=y_,
                    mode='lines',
                    name='Truth',
                    line=dict(color='black', width=1),
                    #  hovertemplate=hovertemplate,
                ),
                row=1,
                col=1)

            # Plot predictions
            fig.add_trace(
                go.Scatter(
                    x=ypred_.index,
                    y=ypred_,
                    mode='lines',
                    name=pred_label,
                    line=dict(color='red', width=1),
                    #  hovertemplate=hovertemplate,
                ),
                row=1,
                col=1)

            # Plot persistence predictions
            fig.add_trace(
                go.Scatter(
                    x=ypred_persistence_.index,
                    y=ypred_persistence_,
                    mode='lines',
                    name=persistence_label,
                    line=dict(color='blue', width=1),
                    #  hovertemplate=hovertemplate,
                ),
                row=1,
                col=1)

            if y_orig is not None:
                fig.add_trace(
                    go.Scatter(
                        x=y_orig[storm_idx].index,
                        y=y_orig[storm_idx],
                        mode='lines',
                        name='Original',
                        line=dict(color='gray', width=1),
                        visible='legendonly',
                    ),
                    row=1,
                    col=1,
                )

            # Add lines for axis
            fig.update_yaxes(showline=True,
                             linewidth=.8,
                             linecolor='black',
                             row=1,
                             col=1)
            fig.update_xaxes(showline=True,
                             linewidth=.8,
                             linecolor='black',
                             row=1,
                             col=1)

            # Plot solar wind parameters
            if sw_to_plot is not None:
                for i in range(n_sw_to_plot):
                    X_sw_i = X_[sw_to_plot[i]]
                    fig.add_trace(
                        go.Scatter(
                            x=X_sw_i.index,
                            y=X_sw_i,
                            mode='lines',
                            name=sw_to_plot[i],
                            line=dict(color='black', width=1),
                            showlegend=False,
                            #  hovertemplate=hovertemplate,
                        ),
                        row=i + 2,
                        col=1)
                    fig.update_yaxes(title_text=sw_to_plot[i],
                                     showline=True,
                                     linewidth=.8,
                                     linecolor='black',
                                     row=i + 2,
                                     col=1)
                    fig.update_xaxes(showline=True,
                                     linewidth=.8,
                                     linecolor='black',
                                     row=i + 2,
                                     col=1)

            fig.update_layout(
                height=650,
                width=850,
                margin=go.layout.Margin(l=30, r=30, b=50, t=50),
                title=dict(
                    text=info,
                    font=dict(size=12),
                ),
                legend=dict(x=0,
                            y=-0.12,
                            orientation='h',
                            xanchor='left',
                            yanchor='bottom',
                            font=dict(size=10, )),
                template="plotly_white",
                hovermode="x unified",
                annotations=[
                    # Show features used.
                    dict(x=0,
                         y=1.03,
                         showarrow=False,
                         xref="paper",
                         yref="paper",
                         text=features_text,
                         font=dict(size=10, ))
                ])
            return fig, None

        else:
            if sw_to_plot is not None:
                fig, ax = plt.subplots(
                    nrows=n_sw_to_plot + 1,
                    ncols=1,
                    sharex=True,
                    figsize=figsize,
                    gridspec_kw={'height_ratios': [4] + [1] * n_sw_to_plot})
                ax0 = ax[0]
            else:
                fig, ax = plt.subplots(sharex=True, figsize=figsize)
                ax0 = ax

            # Plot truth
            # y_.unstack(level=self.storm_level).plot(label='Truth', color='black', linewidth=0.5, ax=ax0)
            ax0.plot(y[storm_idx], label='Truth', color='black', linewidth=0.5)

            # Plot predictions

            # ypred_.unstack(level=self.storm_level).plot(
            #     label=pred_label, color='red', linewidth=0.5, ax=ax0)
            ax0.plot(ypred[storm_idx],
                     label=pred_label,
                     color='red',
                     linewidth=0.5)

            if lower is not None and upper is not None:
                ax0.fill_between(
                    lower.index.get_level_values(level=self.time_level),
                    lower[storm_idx],
                    upper[storm_idx],
                    alpha=0.25,
                    color='red',
                )

            if ypred_persistence is not None:
                # ypred_persistence.unstack(level=self.storm_level).plot(
                #     label=persistence_label, color='blue', linestyle='--',
                #     linewidth=0.5)
                ax0.plot(ypred_persistence[storm_idx],
                         label=persistence_label,
                         color='blue',
                         linestyle='--',
                         linewidth=0.5)

            ax0.legend()
            # Adjust time scale
            locator = mdates.AutoDateLocator(minticks=min_ticks)
            formatter = mdates.ConciseDateFormatter(locator)
            ax0.xaxis.set_major_locator(locator)
            ax0.xaxis.set_major_formatter(formatter)

            if time_range is not None:
                time_range_ = [pd.to_datetime(t) for t in time_range]
                ax0.set_xlim(time_range_)

            if display_info:
                ax0.set_title(info)

            if sw_to_plot is not None:

                for i in range(n_sw_to_plot):
                    ax[i + 1].plot(X[sw_to_plot[i]][storm_idx],
                                   label=sw_to_plot[i],
                                   color='black',
                                   linewidth=0.5)
                    ax[i + 1].legend()
                    ax[i + 1].xaxis.set_major_locator(locator)
                    ax[i + 1].xaxis.set_major_formatter(formatter)

                    if time_range is not None:
                        ax[i + 1].set_xlim(time_range_)

                fig.tight_layout()

            plt.figtext(0.05,
                        0.008,
                        features_text,
                        ha='left',
                        fontsize=12,
                        fontweight='demibold')

            return fig, ax

    def plot_predict(self,
                     y,
                     ypred,
                     X=None,
                     upper=None,
                     lower=None,
                     plot_persistence=True,
                     storms_to_plot=None,
                     display_info=False,
                     figsize=(15, 10),
                     save=True,
                     file_name='prediction_plot.pdf',
                     model_name=None,
                     sw_to_plot=None,
                     time_range=None,
                     min_ticks=10,
                     interactive=False,
                     y_orig=None,
                     **more_info):
        """Plotting method for predictions

        Parameters
        ----------
        y : pd.Series
            Original target
        ypred : pd.Series
            Predicted y
        X : pd.DataFrame, optional
            Dataframe containing features to be plotted, by default None
        upper : pd.Series, optional
            Upper level for prediction, by default None
        lower : pd.Series, optional
            Lower level for prediction, by default None
        plot_persistence : bool, optional
            If True, plot line for prediction for persistence model, by default True
        storms_to_plot : array-like, optional
            Storms to plot, by default None (plot all)
        display_info : bool, optional
            If True, display information about predictions in title, by default False
        figsize : tuple, optional
            figsize in plt.plot, by default (15, 10)
        save : bool, optional
            If True, save plot, by default True
        file_name : str, optional
            Name of file that plot is saved to, by default 'prediction_plot.pdf'
        model_name : str, optional
            Name of model to shown, by default None
        sw_to_plot : list of str, optional
            Names of solar wind parameters to plot, by default None
        time_range : list of str or pd.Timestamp, optional
            List of length 2 containing start and end time to plot, by default None
        min_ticks : int, optional
            min_ticks in mdates.AutoDateLocator, by default 10
        interactive : bool, optional
            If True, use plot_ly, by default False
        y_orig : pd.Series, optional
            Original unsmoothed y, by default None

        Returns
        -------
        None

        Raises
        ------
        ValueError
            X is not specified and plot_persistence is True
        """
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
                raise ValueError(
                    "X needs to specified if plot_persistence is True.")

            ypred_persistence = self._predict_persistence(
                y, Vx=X[self.vx_colname])
        else:
            ypred_persistence = None

        if interactive:
            file_name = _check_file_name(file_name, ['html'])
        else:
            file_name = _check_file_name(file_name, ['pdf'])

        if save and not interactive:
            pdf = PdfPages(file_name)

        for storm in storms_to_plot:
            # rmse = self.score(X.loc[storm], y.loc[storm])
            fig, _ = self._plot_one_storm(storm,
                                          y,
                                          ypred,
                                          X=X,
                                          ypred_persistence=ypred_persistence,
                                          lower=lower,
                                          upper=upper,
                                          display_info=display_info,
                                          figsize=figsize,
                                          model_name=model_name,
                                          sw_to_plot=sw_to_plot,
                                          time_range=time_range,
                                          min_ticks=min_ticks,
                                          interactive=interactive,
                                          y_orig=y_orig,
                                          **more_info)

            if not interactive:
                if save:
                    pdf.savefig(fig)
                    pdf.close()
                else:
                    plt.show()
            else:
                if save:
                    fig.write_html(file_name)

        return None

    def save(self, file_name, valid_ext=['pkl'], **kwargs):
        """Save object

        Parameters
        ----------
        file_name : str
            Name of file
        valid_ext : list, optional
            List of valid extensions, by default ['pkl']

        Returns
        -------
        None
        """

        file_name = _check_file_name(file_name, valid_ext=valid_ext)
        joblib.dump(self, file_name, **kwargs)
        return None


class ARXFeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 auto_order=0,
                 exog_order=2,
                 include_interactions=False,
                 interactions_degree=2,
                 seasonality=False,
                 time_resolution='5T',
                 transformer=None,
                 fit_transformer=True,
                 **transformer_params):
        self.auto_order = auto_order
        self.exog_order = exog_order
        self.include_interactions = include_interactions
        self.interactions_degree = interactions_degree
        self.seasonality = seasonality
        self.time_resolution = time_resolution
        self.transformer = transformer
        self.fit_transformer = fit_transformer
        self.transformer_params = transformer_params

    def fit(self,
            X,
            y=None,
            target_column=None,
            storm_level=0,
            time_level=1,
            keep_pd=False,
            **transformer_fit_params):

        if self.transformer is not None:
            self.transformer = self.transformer.set_params(
                **self.transformer_params)

            if self.include_interactions:
                self.transformer = make_pipeline(
                    PolynomialFeatures(degree=self.interactions_degree,
                                       interaction_only=True,
                                       include_bias=False), self.transformer)
        elif self.include_interactions:
            self.transformer = PolynomialFeatures(
                degree=self.interactions_degree,
                interaction_only=True,
                include_bias=False)

        self.target_column_ = target_column
        self.storm_level_ = storm_level
        self.time_level_ = time_level

        # convert auto_order, exog_order to time steps
        # time_res_minutes = to_offset(self.time_resolution).delta.seconds / 60
        # self.auto_order_timesteps_ = np.rint(self.auto_order /
        #                                      time_res_minutes).astype(int)
        # self.exog_order_timesteps_ = np.rint(self.exog_order /
        #                                      time_res_minutes).astype(int)

        if self.transformer is not None:
            self.transformer.set_params(**self.transformer_params)

            if self.fit_transformer:
                self.transformer = _pd_fit(self.transformer, X,
                                           **transformer_fit_params)

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
                "X does not have regular time increments with the specified time resolution."
            )

        if self.target_column_ is None:
            y_ = None
            X_ = X.to_numpy()
        else:
            y_ = X.iloc[:, self.target_column_].to_numpy()
            X_ = X.drop(columns=X.columns[self.target_column_]).to_numpy()

        # TODO: write my own version
        p = MetaLagFeatureProcessor(X_, y_, self.auto_order,
                                    [self.exog_order] * X_.shape[1],
                                    [0] * X_.shape[1])

        features = p.generate_lag_features()

        if self.seasonality:
            yr_term = ((2 * np.pi * times.dayofyear) / 365).to_numpy().reshape(
                -1, 1)
            day_term = ((2 * np.pi * times.hour) / 24).to_numpy().reshape(
                -1, 1)
            sin_yr = np.sin(yr_term)
            cos_yr = np.cos(yr_term)
            sin_day = np.sin(day_term)
            cos_day = np.cos(day_term)
            features = np.concatenate(
                (features, sin_yr, cos_yr, sin_day, cos_day), axis=1)

        return features


class PropagationTimeProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, Vx=None, time_resolution='5T', D=1500000, lazy=True):
        self.Vx = Vx
        self.time_resolution = time_resolution
        self.D = D
        self.lazy = lazy

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

    def _compute_propagated_time(self, x):
        return (x['time'] + pd.Timedelta(x['prop_time'], unit='sec')).floor(
            freq=self.time_resolution)

    def _compute_times(self, storm_level=0, time_level=1):
        propagation_in_sec_ = self.D / self.Vx.abs()
        propagation_in_sec_.rename("prop_time", inplace=True)

        if isinstance(propagation_in_sec_.index, pd.MultiIndex):
            propagation_in_sec_.index.rename(names='time',
                                             level=time_level,
                                             inplace=True)
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

        self._compute_times(storm_level=storm_level, time_level=time_level)
        self.mask_ = self._compute_mask(X, time_level=self.time_level_)

        return self

    def transform(self, X, y=None):
        check_is_fitted(self)

        if self.Vx is not None:
            if isinstance(X.index, pd.MultiIndex):
                storms = X.index.get_level_values(level=self.storm_level_)
                return X.reindex(
                    [storms[self.mask_], self.propagated_times[self.mask_]])
            else:
                return X.reindex([self.mask_])
        else:
            return X


class TargetProcessor(BaseEstimator, TransformerMixin):
    def __init__(
            self,
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

    def fit(self,
            X,
            y=None,
            storm_level=0,
            time_level=1,
            **transformer_fit_params):

        self.storm_level_ = storm_level
        self.time_level_ = time_level

        # Transform X if transformer is provided
        if self.transformer is not None:
            self.transformer.set_params(**self.transformer_params)
            self.transformer = _pd_fit(self.transformer, X,
                                       **transformer_fit_params)

        # Get future times to predict
        self.times_to_predict_ = X.index.get_level_values(
            level=self.time_level_) + pd.Timedelta(minutes=self.pred_step)
        self.mask_ = self._get_mask(X)

        return self

    def transform(self, X, y=None):
        check_is_fitted(self)

        X_ = _pd_transform(self.transformer, X)

        storms = X.index.get_level_values(level=self.storm_level_)
        X_ = X_.reindex([storms, self.times_to_predict_])

        return X_

    def _get_mask(self, X):
        X_times = X.index.get_level_values(level=self.time_level_)

        # Get mask of where times_to_predict are in X's time index
        mask = np.in1d(self.times_to_predict_, X_times)

        return mask
