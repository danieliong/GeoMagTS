import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics.regression import r2_score, mean_squared_error
from geomag_utils import get_storm_indices, create_narx_model
from sklearn.model_selection import GridSearchCV

import sys
sys.path.insert(1, '../../fireTS')
from fireTS.utils import shift
from fireTS.core import GeneralAutoRegressor
from fireTS.models import DirectAutoRegressor

class GeoMagAutoRegressor(GeneralAutoRegressor):
    def __init__(self,
                 base_estimator,
                 auto_order,
                 exog_order,
                 pred_step,
                 exog_delay=None,
                 **base_params):
        super(GeneralAutoRegressor, self).__init__(
            base_estimator,
            auto_order,
            exog_order,
            exog_delay=exog_delay,
            pred_step=pred_step,
            scaleX =True,
            scaley=True,
            **base_params)
    
    def set_scalerX(self, scaler):
        self.scalerX = scaler

    def set_scaler_y(self, scaler):
        self.scaler_y = scaler
    



class NARXNN(DirectAutoRegressor):
    def __init__(self,
                 auto_order,
                 exog_order,
                 pred_step,
                 exog_delay=None,
                 n_hidden=18,
                 learning_rate=0.0001,
                 scaleX=True,
                 scaley=True,
                 **base_params):
    
        base_estimator = KerasRegressor(build_fn=create_narx_model, n_hidden=n_hidden, learning_rate=learning_rate)
        super(NARXNN, self).__init__(
            base_estimator,
            auto_order,
            exog_order,
            exog_delay=exog_delay,
            pred_step=pred_step,
            **base_params)
        
        self.scaleX = scaleX
        if self.scaleX:
            self.scaler_X = RobustScaler()
        self.scaley = scaley
        if self.scaley:
            self.scaler_y = RobustScaler()
    
    def set_scalerX(self, scaler):
        self.scalerX = scaler 
    def set_scaler_y(self, scaler):
        self.scaler_y = scaler 
    
    def _preprocess_data_bystorms(self, X, y, indices_list):        
        preprocessed_data = list(
            zip(*[self._preprocess_data(X[indices], y[indices]) 
                  for indices in indices_list])
            )

        features = np.vstack(preprocessed_data[0])
        target = np.concatenate(preprocessed_data[1])
        return features, target
    
    def fit_bystorms(self, X, y, indices_list, **params):
        if self.scaleX:
            X_storms = np.vstack([X[indices] for indices in indices_list])
            self.scaler_X = self.scaler_X.fit(X_storms)
            X = self.scaler_X.transform(X)
        if self.scaley:
            y_storms = np.hstack([y[indices] for indices in indices_list])
            self.scaler_y = self.scaler_y.fit(y_storms.reshape(-1,1))
            y = self.scaler_y.transform(y.reshape(-1,1)).flatten()
        features, target = self._preprocess_data_bystorms(X, y, indices_list)
        self.base_estimator.fit(features, target, **params)
    
    def grid_search_bystorms(self, X, y, indices_list, para_grid, **params):
        grid = GridSearchCV(self.base_estimator, para_grid, **params)        
        if self.scaleX:
            X_storms = np.vstack([X[indices] for indices in indices_list])
            self.scaler_X = self.scaler_X.fit(X_storms)
            X = self.scaler_X.transform(X)
        if self.scaley:
            y_storms = np.hstack([y[indices] for indices in indices_list])
            self.scaler_y = self.scaler_y.fit(y_storms.reshape(-1, 1))
            y = self.scaler_y.transform(y.reshape(-1, 1)).flatten()
        features, target = self._preprocess_data_bystorms(X, y, indices_list)
        grid.fit(features, target)
        self.base_estimator.set_params(**grid.best_params_)
    
    def predict_scaled(self, X, y):
        if self.scaleX:
            X = self.scaler_X.transform(X)
        if self.scaley:
            y = self.scaler_y.transform(y.reshape(-1,1)).flatten()
        y_pred = self.predict(X, y)
        if self.scaley:
            return self.scaler_y.inverse_transform(y_pred.reshape(-1,1)).flatten()
        else:
            return y_pred
    
    def score_shifted(self, X, y, method='mse', verbose=False):
        ypred = self.predict_scaled(X, y)
        y_shifted = shift(y, self.pred_step)
        mask = np.isnan(y_shifted) | np.isnan(ypred)
        if verbose:
            print('Evaluating {} score, {} of {} data points are evaluated.'.
                format(method, np.sum(~mask), y.shape[0]))
        if method == "r2":
            return r2_score(y_shifted[~mask], ypred[~mask])
        elif method == "mse":
            return mean_squared_error(y_shifted[~mask], ypred[~mask])
        else:
            raise ValueError(
                '{} method is not supported. Please choose from \"r2\" or \"mse\".')
    
    def plot_predict(self, X_test, y_test, 
                     dates=None, display_info=False, 
                     **plt_params):
        y_pred = self.predict_scaled(X_test, y_test)
        rmse = np.sqrt(self.score(X_test, y_test, method='mse'))

        # y_shifted = shift(y_test, self.pred_step)

        fig, ax = plt.subplots(sharex=True, figsize=(10, 7), **plt_params)
        if dates is None:
            ax.plot(y_test, label='Truth', color='black', linewidth=0.5)
            ax.plot(y_pred, 
                    label=str(self.pred_step)+'-step ahead prediction', color='red', linewidth=0.5)
        else:
            ax.plot(dates, y_test, 
                    label='Truth', color='black',linewidth=0.5)
            ax.plot(dates, y_pred, 
                    label=str(self.pred_step)+'-step ahead prediction', color='red', linewidth=0.5)
        ax.legend()
        if display_info:
            ax.set_title(
                'n_hidden='+str(self.base_estimator.get_params()['n_hidden'])+', '+
                'learn_rate='+str(self.base_estimator.get_params()['learning_rate'])+', '+
                'auto_order='+str(self.auto_order)+', '+
                'exog_order='+str(self.exog_order[0])+', '+
                'RMSE='+str(np.round(rmse, decimals=2)))
        return fig, ax
        
