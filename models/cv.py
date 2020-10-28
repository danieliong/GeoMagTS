
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import sparse
from joblib import Parallel, delayed, effective_n_jobs
from typing import Optional, TypeVar, Tuple
from collections import namedtuple
# from functools import cached_property

from sklearn.base import MultiOutputMixin, clone
from sklearn.model_selection import check_cv, ParameterGrid
from sklearn.utils import check_array
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.metrics import mean_squared_error

from .additive import SparseAdditiveRegression

# Adapted from sklearn's _path_residuals function     
array_like = TypeVar('array_like', np.ndarray, list, tuple)

path_result = namedtuple('path_result', ['hyperparams','score'])

# Adapted from sklearn's LinearModelCV base class
# Link: https: // github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/linear_model/_coordinate_descent.py
class BaseTuning(MultiOutputMixin, metaclass=ABCMeta):
    """Base class for iterative model fitting along a regularization path"""

    def __init__(self, 
                #  n_hyperparams:int=100, 
                 hyperparams:Optional[dict]=None, *, 
                 cv=None, # What is the type?
                 verbose_cv:bool=False, 
                 n_jobs:Optional[int]=None, 
                 random_state:Optional[int]=None,
                 clone_model=True,
                 mse:bool=True, 
                 refit:bool=True,
                 **model_kwargs):
        
        # self.n_hyperparams = n_hyperparams
        self.hyperparams = hyperparams
        self.cv = cv
        self.verbose_cv = verbose_cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.clone_model = clone_model
        self.mse = mse
        self.refit = refit
        # self.model_kwargs = model_kwargs
        
        self.model = self.model(**model_kwargs)
    
    # Check if model and params defined in subclass is valid.
    def __init_subclass__(cls, *args, **kwargs) -> None:
        super().__init_subclass__(*args, **kwargs)
        
        valid_model = (isinstance(cls.model, type) and 
                       (hasattr(cls.model, '_fit') or 
                        hasattr(cls.model, 'fit'))) 
        if not valid_model:
            raise ValueError(
                "model must be of type 'type' and have either a _fit and/or fit method.")
        
        def valid_params(params):
            return (isinstance(params, tuple) and
                    (all([type(p) == str for p in params])))
        
        if not valid_params(cls.param_names):
            raise TypeError("params must be a tuple of str.")
        
        # if not valid_params(cls.hyperparam_names):
        #     raise TypeError("hyperparams must be a tuple of str.")
    
    @property
    @abstractmethod
    def model(self) -> type:
        pass 
    
    # Define tuple of names of parameters
    @property
    @abstractmethod
    def param_names(self) -> Tuple[str, ...]:
        pass
    
    @property
    def hyperparam_names(self):
        return self.hyperparams.keys()
    
    @property  
    def hyperparams_grid_(self):
        return ParameterGrid(self.hyperparams)            
    
    def _compute_score_path(
        self,
        X:np.ndarray,
        y:np.ndarray,
        train:np.ndarray, 
        test:np.ndarray,
        **fit_kwargs):
        
        # Check if keys in hyperparams are in model
        keys_in_model = [hasattr(self.model, name) 
                         for name in self.hyperparam_names]
        if not all(keys_in_model):
            raise ValueError(
                "All keys in hyperparams dict must be parameters in model.")
        
        if self.clone_model:
            model = clone(self.model)
        else:
            model = self.model
        
        # hyperparams_grid = ParameterGrid(self.hyperparams)
        
        n_grid = len(self.hyperparams_grid_)
        scores = np.zeros(n_grid)
        
        for i in range(n_grid):
        # for hyperparams in self.hyperparams_grid_:
            hyperparams = self.hyperparams_grid_[i]
            for name in self.hyperparam_names:
                setattr(model, name, hyperparams[name])
            
            model.fit(X[train], y[train], **fit_kwargs)
            scores[i] = self._score(model, X[test], y[test], 
                                              mse=self.mse)
            
            # yield path_result(hyperparams, score)
        return scores 


    def predict(self, X, y=None, **kwargs):
        if hasattr(self.model, 'predict'):
            return self.model.predict(X, y, **kwargs)
        else:
            raise AttributeError('model does not have a predict method.')
    
    @staticmethod
    def _score(model, X, y, mse=True, **kwargs):
        
        if mse or not hasattr(model, 'score'):
            ypred = model.predict(X)
            return mean_squared_error(y, ypred, **kwargs)
        
        if hasattr(model, 'score'):
            return model.score(X, y, **kwargs)
    
    def score(self, X, y, mse=True, **kwargs):
        return self._score(self, X, y, mse=mse, **kwargs)
    
    def fit(self, X, y=None, *, groups=None, **fit_kwargs):
        
        cv = check_cv(self.cv)
        
        jobs = (delayed(self._compute_score_path)(
            X, y, train, test, **fit_kwargs) 
                for train, test in cv.split(X, y, groups=groups)) 
        
        score_path = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose_cv,
            **_joblib_parallel_args(prefer='threads'))(jobs)
        
        self.mean_scores_ = np.mean(score_path, axis=0)
        
        self.best_index_ = np.argmin(self.mean_scores_)
        self.best_score_ = np.min(self.mean_scores_)
        self.best_hyperparams_ = self.hyperparams_grid_[self.best_index_]
        
        for name in self.hyperparam_names:
            setattr(self.model, name, self.best_hyperparams_[name])
        
        self.model.fit(X, y, **fit_kwargs)
        
        return self
    

class SparseAdditiveRegressionTuning(BaseTuning):
    model = SparseAdditiveRegression
    param_names = ('intercept_', 'fhat_')
    # hyperparam_names = ('lam')
    
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
        
        
