import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model._coordinate_descent import LinearModelCV 
from sklearn.metrics import mean_squared_error
# from sklearn.linear_model._coordinate_descent import path_residuals
from sklearn.utils import check_array, check_random_state
from functools import partial
import time
from typing import Optional, Tuple
from collections import namedtuple
import warnings
# from itertools import filterfalse
from collections import deque

# import sys
# sys.path.insert(0, '..')
from ..glmgen import trendfilter, predict_tf, poly_coefs_tf, falling_fact_coefs_tf, maxlam_tf  
# from .cv import AdditiveModelCV


def t_loss(y, f_hat, intercept, df=10):
    # TODO: Need to figure out how to estimate scale parameter.
    pass

def squared_loss(y: np.ndarray, 
                 fhat: np.ndarray, 
                 intercept:float) -> np.float64:
    y_fitted = np.sum(fhat, axis=-1) + intercept 
    return np.mean((y - y_fitted)**2)/2

def gaussian_grad(y: np.ndarray, 
                  fhat: np.ndarray, 
                  intercept:float) -> np.ndarray:
    y_fitted = np.sum(fhat, axis=-1) + intercept
    n = y.shape[0]
    return -(1/n) * (y - y_fitted) 
 
 
TF_DEFAULT_KWARGS = dict(
    family=0,
    max_iter=500, 
    lam_flag=1,
    nlambda=1, 
    lambda_min_ratio=1e-3, 
    rho=1, 
    obj_tol=1e-6, 
    obj_tol_newton=1e-6, 
    max_iter_newton=50,
    x_tol=1e-6, 
    alpha_ls=0.5, 
    gamma_ls=0.9, 
    max_iter_ls=20,
    thinning=1, 
    verbose=0)


def _admm_tf(x:np.ndarray, 
             y:np.ndarray, *, 
             lam1:float, 
             lam2:float, 
             w:Optional[np.ndarray]=None, 
             order:int=2,
             **tf_kwargs) -> Tuple[np.ndarray, float]:
    
    admm_tf_res = namedtuple('admm_tf_res', ['f', 'thres'])
    
    q1_x = np.percentile(x, 25, interpolation='midpoint')
    q3_x = np.percentile(x, 75, interpolation='midpoint')
    iqr_x = (q3_x - q1_x)
    diff_x = np.max(x) - np.min(x)
    
    if 'x_tol' in tf_kwargs.keys():
        tf_kwargs['x_tol'] = tf_kwargs['x_tol'] * max(
            iqr_x, (diff_x/2))
    else:
        tf_kwargs['x_tol'] = 1e-6 * max(
            iqr_x, (diff_x/2))
    
    n = y.shape[0]
    
    # Order x, y here
    ord = np.argsort(x)
    x = np.sort(x)
    y = y[ord]

    if w is None:
        tf = trendfilter(x_ord=x,
                         y_ord=y,
                         w_ord=np.ones(n),
                         k=order,
                         lam=np.array([lam2]).astype(np.float),
                         **tf_kwargs)
    else:
        w = w[ord]
        tf = trendfilter(x_ord=x,
                         y_ord=y,
                         w_ord=w,
                         k=order,
                         lam=np.array([lam2]).astype(np.float),
                         **tf_kwargs)
    # del y, w

    f_hat = tf['f_hat'][0, :]
    x_thinned = tf['x']
    # del tf
    
    converged = (tf['status'] == 0)
    
    if tf['status'] == 1:
        warnings.warn("Trend filter estimates did not converge.")

    if x_thinned.shape[0] != n:
        f_hat = np.interp(x, xp=x_thinned, fp=f_hat)
    # del x_thinned
    
    # Reorder f_thres to match original order
    f_new = np.zeros(n)
    f_new[ord] = f_hat

    return f_new, converged
    # return f_thres[ord], thres
 
class SparseAdditiveRegression(BaseEstimator,  RegressorMixin):
    def __init__(self, 
                 family: str ='gaussian',
                 method: str ='tf',
                 order: int = 2,
                 lam: Optional[float] = 1.0,
                 init_step_size: Optional[float] = None,
                 init_fhat: Optional[np.ndarray] = None,
                 init_intercept: Optional[float] = None,
                 max_iter:int = 200,
                 tol: float = 1e-4,
                 alpha: float = 0.5, # For line search
                 warm_start: bool = True,
                 verbose: bool = False,
                #  copy_X=True,
                 **tf_kwargs):
        
        self.family = family
        self.method = method
        self.order = order
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.warm_start = warm_start
        self.verbose = verbose
        self.tf_kwargs = dict(TF_DEFAULT_KWARGS)
        self.tf_kwargs.update(**tf_kwargs)
        
        self.lam = lam
        self.init_fhat = init_fhat
        self.init_intercept = init_intercept
        self.init_step_size = init_step_size
    
    parameters = namedtuple('parameters', ['intercept_', 'fhat_'])
    
    prox_grad_results = namedtuple(
        'prox_grad_res', ['intercept_', 'fhat_', 'active_mask_', 'step_'])
    
    fit_result = namedtuple(
        'fit_result', ['intercept_', 'fhat_', 'active_mask_',
                       'step_', 'n_iter_'])
    
    # TODO: Validation for loss, method
    
    # @property
    # def init_params(self) -> Tuple[float, np.ndarray]:
    #     return self.init_intercept, self.init_fhat
    
    @property
    def lam1(self) -> float:
        return self.lam
    
    @property 
    def lam2(self) -> float:
        return self.lam**2
    
    def loss(self):
        pass
    
    @property
    def p_(self):
        return self.active_mask_.shape[0]
    
    @property
    def p_active_(self):
        return np.sum(self.active_mask_)
    
    # GetZ
    # fhat: n*p matrix
    # @staticmethod
    # @classmethod
    def _prox_grad_fixed_step(self,
        X: np.ndarray, 
        grad: np.ndarray, 
        w:Optional[np.ndarray]=None, *,
        # f_hat: np.ndarray, 
        # intercept: float, 
        # lam1: float, 
        # lam2: float, 
        step: float, 
        # order:int=2, 
        # method:str='tf',
        # tf_kwargs:Optional[dict]=None
        ) -> Tuple[float, np.ndarray, np.ndarray]:
        
        start_time = time.time()
        
        n, p = self.fhat_.shape[0], self.active_mask_.shape[0]
        
        intercept_ = self.intercept_ - (step * sum(grad))
        
        active_mask = np.array([False]*p) 
        
        # # Convert f_hat to iterator over columns
        # f_hat_iter = np.nditer(f_hat, flags=['external_loop'], 
        #                        op_flags=['readonly'], order='F')
        # Create iterator for computing intermediate step
        inter_step = ((f - step*grad) for f in self.fhat_.T)
        # del f_hat
        
        step_grad = step*grad
        del grad
        
        def intermed_step():
            fhat_idx = iter(range(self.p_active_))
            for j in range(p):
                if self.active_mask_[j]:
                    inter_step = self.fhat_[:,next(fhat_idx)] - step_grad
                else:
                    inter_step = -step_grad
                    
                yield inter_step - np.mean(inter_step)
                    
        inter_step = intermed_step()
        fhat_ = deque()
                
        for j, inter_step_j in enumerate(inter_step):
        # for j, (inter_step_j, x_ord, ord) in enumerate(zip(inter_step, *X_it)):
            # y_ord = inter_step[ord] - np.mean(inter_step)
            if self.verbose:
                print("[PG] j = "+str(j))
                
            if self.method == 'tf':
                lam1 = (self.lam1*step)/n
                lam2 = (self.lam2*step)
                
                order = np.argsort(X[:,j])
                max_lam = maxlam_tf(
                    x_ord=np.sort(X[:,j]),
                    y_ord=inter_step_j[order],
                    w_ord=(np.ones(n) if w is None else w[order,j]),
                    k=self.order
                    )
                
                if lam2 > max_lam:
                    step_new = (max_lam / self.lam2)
                    warnings.warn(
                        "step*self.lam2 > max_lam = "+str(max_lam)+".")
                    # converged = False
                    return intercept_, self.fhat_, self.active_mask_, step_new
                
                start_time_admm = time.time()
                
                f_j, converged = _admm_tf(
                    x=X[:,j], y=inter_step_j, 
                    w=(None if w is None else w[:,j]),
                    lam1=lam1,
                    lam2=lam2,
                    order=self.order,
                    **self.tf_kwargs
                    )
                
                if self.verbose:
                    run_time_admm = time.time() - start_time_admm
                    print("[ADMM] Run time: {: .0f} seconds.".format(
                        run_time_admm
                    ))
                
                f_hat_norm = np.sqrt(np.mean(f_j**2))
                if f_hat_norm < 1e-100:
                    warnings.warn("f_hat_norm < 1e-100.")
                
                thres = max(1 - (lam1 / (f_hat_norm+1e-30)), 0)
                if thres != 0:
                    active_mask[j] = True
                    fhat_.append(thres*f_j)
                    
                if not converged:
                    warnings.warn("ADMM estimates did not converge.")
        
        fhat_ = np.array(fhat_).T
        
        if self.verbose:
            run_time = time.time() - start_time
            print("[PG] Run time: {: .0f} seconds.".format(run_time))
        
        return intercept_, fhat_, active_mask, step
        # res = namedtuple('res', ['intercept','f','active_mask'])
        # return cls.prox_grad_results(intercept_, f_hat_, active_mask, None)
        # return intercept, f_hat, active_mask    
        
    # LineSearch
    # @classmethod
    def _prox_grad_line_search(
        self, X:np.ndarray, 
        y:np.ndarray, 
        w:Optional[np.ndarray],
        # f_hat:np.ndarray, 
        # intercept:float, 
        step:float,
        # lam1:float, 
        # lam2:float, 
        # alpha:float=0.5, 
        # order:int=2, 
        # family:str='gaussian', 
        # method:str='tf', 
        # verbose:bool=False, 
        # tf_kwargs:Optional[dict]=None
        ) -> Tuple[float, np.ndarray, np.ndarray, float]:
        
        if self.family == 'gaussian':
            loss = squared_loss(y, self.fhat_, self.intercept_)
            grad = gaussian_grad(y, self.fhat_, self.intercept_)
        
        n = y.shape[0]
        
        converged = False
        # step = self.init_step_size
        iter_ = 1
        
        if self.verbose:
            start_time = time.time()
            
        sum_grad = np.sum(grad)
        
        while not converged:
            if self.verbose:
                print(
                    '[LS] Iteration #: '+str(iter_)+", step size = "+str(step)
                    )
            
            # intercept_new, f_new, active_mask = self._prox_grad_fixed_step(
            #     X, grad, f_hat=self.fhat_, intercept=self.intercept_, 
            #     lam1=lam1, lam2=lam2, step=step, w=w, 
            #     order=order, method=method, 
            #     tf_kwargs=tf_kwargs)
            
            intercept_new, f_new, active_mask, step_ = \
                self._prox_grad_fixed_step(
                    X, grad, w, step=step)
            
            if step_ != step:
                step = step_
                if step_ == 1e-100:
                    warnings.warn(
                        "Updated step < 1e-100. Line search will be stopped.")
                    converged = False
                    self.status_ = "Updated step < 1e-100 [max_lam small]."
                    return converged, iter_, self.intercept_, self.fhat_, self.active_mask_ 
                
                warnings.warn("Line search will restart with step = max_lam / self.lam2 = "+str(step_)+".")
                continue 
            
            if self.family == 'gaussian':
                lhs = squared_loss(y, f_new, intercept_new)
            
            if not np.isfinite(lhs):
                warnings.warn("New loss is not finite.") 
                # step = self.alpha * step
                converged = False
                self.status = "Updated loss not finite." 
                # continue
                return converged, iter_, self.intercept_, self.fhat_, self.active_mask_ 
                
            sum_inner_prod = 0
            # In case predictors are all inactive
            if active_mask.any():
                sum_inner_prod = sum_inner_prod + np.dot(grad, f_new).sum()
            if self.active_mask_.any():
                sum_inner_prod = sum_inner_prod - np.dot(
                    grad, self.fhat_).sum()
            
            p_active_new = np.sum(active_mask) 
            new_i = iter(range(p_active_new))
            old_i = iter(range(self.p_active_))
            
            norm_sq = 0
            # Conserves memory
            for i in range(self.p_):
                if active_mask[i] and self.active_mask_[i]:
                    norm_sq = norm_sq + np.sum(
                        (f_new[:,next(new_i)] - self.fhat_[:,next(old_i)])**2
                        )
                elif active_mask[i] and not self.active_mask_[i]:
                    norm_sq = norm_sq + np.sum(f_new[:,next(new_i)]**2)
                elif not active_mask[i] and self.active_mask_[i]:
                    norm_sq = norm_sq + np.sum(self.fhat_[:,next(old_i)]**2)
                
                if not np.isfinite(norm_sq):
                    break 
            # norm_sq = norm_sq / n 
            
            if step < 1e-100:
                rhs = np.inf
            else:
                rhs = loss + (sum_grad * (intercept_new - self.intercept_)) + \
                    sum_inner_prod + \
                        (1/(2*step)) * (norm_sq + \
                                        (intercept_new - self.intercept_)**2)
            
            if not np.isfinite(rhs):
                warnings.warn(
                    "Right hand side of majorizing inequality is not finite.")
                # step = self.alpha * step
                converged = False
                self.status_ = "RHS not finite."
                return converged, iter_, self.intercept_, self.fhat_, self.active_mask_
            
            if self.verbose:
                # if np.isinf(lhs) or np.isinf(rhs):
                #     diff = 0 
                # else:
                #     diff = (lhs - rhs) 
                diff = (lhs - rhs)
                print("[LS] LHS - RHS = "+str(diff))
            
            if lhs <= rhs:
                converged = True
                # self.fhat_ = f_new
                # self.intercept_ = intercept_new
                # self.active_mask_ = active_mask
                
                if self.verbose:
                    run_time = time.time() - start_time
                    print('[LS] Converged. Run time: {:.0f} seconds.'.format(run_time))
                
                # return converged, iter_, intercept_new, f_new, active_mask
                
            else:
                step = self.alpha * step
                
                if step < 1e-100:
                    warnings.warn("[Line Search] step < 1e-100.")
                    converged = False
                    self.status_ = "Updated step < 1e-100 [LS didn't converge]."
                    return converged, iter_, self.intercept_, self.fhat_, self.active_mask_ 
                    # return False, iter_, intercept_new, f_new, active_mask
                
                if self.verbose:
                    print('[LS] Updated step size: '+str(step))
                iter_ = iter_ + 1
        
        # return cls.prox_grad_results(intercept_new, f_new, active_mask, step)
        # return converged
        return converged, iter_, intercept_new, f_new, active_mask  
    
    # Change back to instance method
    # @classmethod
    # init values: intercept, fhat, step
    def _fit(self, 
             X:np.ndarray, 
             y:np.ndarray, 
            #  lam1:float, 
            #  lam2:float, 
            #  init_fhat:Optional[np.ndarray]=None, 
            #  init_intercept:Optional[np.ndarray]=None, 
             w: Optional[np.ndarray]=None, 
             step:Optional[float]=None, 
            #  alpha:float=0.5, 
            #  order:float=2, 
            #  family:str='gaussian', 
            #  method:str='tf', 
            #  max_iter:int=100, 
            #  tol:float=1e-4, 
            #  verbose:bool=False,
            #  tf_kwargs:Optional[dict]=None
             ) -> Tuple[float, np.ndarray, np.ndarray, float]:
        
        iter_ = 1
        converged = False
        
        # f_old = self.init_fhat
        # intercept_old = self.init_intercept
        # del init_fhat, init_intercept

        if self.verbose:
            start_time = time.time()

        self.ls_status_ = [] 
        self.ls_n_iters_ = []
        # self.rel_changes_ = []
        self.loss_ = [np.inf]
        
        while not converged and iter_ <= self.max_iter:
            if self.verbose:
                print("[Fit] Iteration #: " + str(iter_))
                # print("------------------------")

            # intercept_new, f_new, active_mask, step = \
            # intercept_new, f_new, active_mask, _ = \
            # self._prox_grad_line_search(X, y, 
            #     w=w, f_hat=f_old, intercept=intercept_old, 
            #     step=step, lam1=lam1, lam2=lam2, alpha=alpha, order=order, family=family, method=method, verbose=verbose, tf_kwargs=tf_kwargs)
            
            ls_converged, ls_iter, intercept_new, f_new, active_mask = \
                self._prox_grad_line_search(
                    X, y, w=w, step=step)
                
            self.ls_status_.append(ls_converged)
            self.ls_n_iters_.append(ls_iter)
            
            if self.family == 'gaussian':
                loss_new = squared_loss(y, f_new, intercept_new)
                
            if ls_converged:
                # numer = np.mean((f_new - self.fhat_)**2) + \
                #     (intercept_new - self.intercept_)**2
                # denom = np.mean(self.fhat_**2) + (self.intercept_**2)
                # rel_change = numer / (denom+1e-30)
                loss_change = self.loss_[-1] - loss_new
            # else:
            #     # rel_change = 0
            #     loss_change = 0
                # warnings.warn(
                #     "Line search in iter #"+str(iter_)+" did not converge.")
                
            self.loss_.append(loss_new)
            
            # self.rel_changes_.append(rel_change)

            if self.verbose:
                # print("[Fit] Relative change: "+str(rel_change))
                print("[Fit] Loss change: "+str(loss_change))
            
            if ls_converged:
                self.intercept_ = intercept_new
                self.fhat_ = f_new
                self.active_mask_ = active_mask
            else:
                warnings.warn("Line search failed to converge. Algorithm will be stopped.")
                converged = False
                # self.status_ = "Line search failed."
                break 
                # self.converged = False
                # self.n_iter_ = 
                # return None
                
            if (loss_change <= self.tol and loss_change > 0):
                converged = True
                if self.verbose:
                    run_time = time.time() - start_time
                    print(
                        "[Fit] Converged. Run time: {:.0f} seconds.".format(run_time))
                self.status_ = "Converged!"   
            else:
                if loss_change <= 0:
                    warnings.warn("Loss did not decrease from iter. "+str(iter_)+" to iter. "+str(iter_+1)+".")
                    
                iter_ = iter_ + 1
                # res_old = res_old
                # intercept_old, f_old = intercept_new, f_new
                
            if iter_ == self.max_iter:
                warnings.warn("Max iter. was reached.")
                self.status_ = "Max iter. reached."
            
        self.converged = converged
        self.n_iter_ = iter_
            
        return None

        # return cls.fit_result(intercept_new, f_new, active_mask, step, iter)
        # return intercept_new, f_new, active_mask, step
    
    def _prefit(self, X, y):
        
        # If not fitted 
        if not hasattr(self, 'fhat_'):
            y_norm = np.sqrt(np.mean(y**2))
            if self.lam >= y_norm:
                warnings.warn("self.lam >= ||y||_n. This will result in f_hat being all zeros.")
        
        if self.warm_start:
            # if hasattr(self, 'fhat_'):
            #     init_fhat = self.fhat_
            # else:
            #     init_fhat = np.zeros(X.shape)
            if not hasattr(self, 'fhat_'):
                self.fhat_ = np.zeros(X.shape)
                self.active_mask_ = np.array([False]*X.shape[1])
            elif not hasattr(self, 'active_mask_'):
                self.active_mask_ = self.fhat_.any(axis=0)
                
            # if hasattr(self, 'intercept_'):
            #     init_intercept = self.intercept_
            # else:
            #     init_intercept = np.mean(y)
            if not hasattr(self, 'intercept_'):
                self.intercept_ = np.mean(y)
        else:
            if self.init_fhat is None:
                self.fhat_ = np.zeros(X.shape)
                self.active_mask_ = np.array([False]*X.shape[1])
            else:
                self.fhat_ = self.init_fhat
                self.active_mask_ = self.fhat_.any(axis=0)

            if self.init_intercept is None:
                self.intercept_ = np.mean(y)
            else:
                self.intercept_ = self.init_intercept

        # if not hasattr(self, 'step_'):
        if self.init_step_size is None:
            self.step_ = y.shape[0] 
        else:
            self.step_ = self.init_step_size
        
        self.status_ = "Not fit yet."
        # return self.parameters(init_intercept, init_fhat)
        return None
            
    # proxGrad_one 
    def fit(self, X: np.ndarray, 
            y: np.ndarray, sample_weights=None):
        
        start_time = time.time()
        
        self._prefit(X, y)
        
        self._fit(X, y, w=sample_weights, step=self.step_)
        
        # init_intercept, init_fhat = self._prefit(X, y)
        
        # fit_res = self._fit(
        #     X=X, y=y, 
        #     init_fhat=init_fhat, 
        #     init_intercept=init_intercept, 
        #     step=self.step_,
        #     w=sample_weights, 
        #     lam1=self.lam1, 
        #     lam2=self.lam2, 
        #     alpha=self.alpha,
        #     order=self.order, 
        #     family=self.family, 
        #     method=self.method,
        #     max_iter=self.max_iter, 
        #     tol=self.tol, 
        #     verbose=self.verbose,
        #     tf_kwargs=self.tf_kwargs)
        
        # for name in self.fit_result._fields:
        #     setattr(self, name, getattr(fit_res, name))
        
        # Only save columns of active predictors
        # self.fhat_ = fhat_[:,self.active_mask_]
        self.X_train_ = X[:,self.active_mask_]
        
        self.fit_time_ = time.time() - start_time
        
        return self 
    
    def _predict_tf(self, x_new, x_orig, beta):
        
        ord = np.argsort(x_orig)
        x_orig = np.sort(x_orig)
        
        return predict_tf(x0=x_new, x=x_orig, beta=beta[ord], 
                          k=self.order, family=0, zero_tol=1e-6)
        
    def predict(self, X, y=None, output='ypred'):
        
        n = X.shape[0]
        
        if output != 'ypred' and output != 'fhat':
            raise ValueError("output = "+str(output) +
                            " is currently not supported.")
        
        p_orig = len(self.active_mask_)
        if X.shape[1] != p_orig:
            raise ValueError("X must have "+str(p_orig)+" columns.")
        
        # Remove inactive predictors
        X = X[:,self.active_mask_]
        
        f0 = np.transpose(
            [self._predict_tf(x_new, x_orig, beta)
             for x_new, x_orig, beta in zip(
                 X.T, self.X_train_.T, self.fhat_.T)
             ])
        
        # Replace NaN with 0.
        f0 = np.nan_to_num(f0)
        
        if output == 'ypred':
            if len(f0) == 0:
                return np.ones(n) * self.intercept_
            else: 
                return np.sum(f0, axis=-1) + self.intercept_
        elif output == 'fhat':
            return f0
        
        # n,p = X.shape
        
        # falling_fact_basis_vec = np.vectorize(
        #     falling_fact_basis, exclude=['x','k'])
        
        # f0 = np.zeros((n,p))
        # for j in range(p):
        #     if self.active_mask_[j]:
        #         h = falling_fact_basis(X)
        #         f0[:,j] =                   

    def score(self, X, y, **kwargs):
        ypred = self.predict(X, output='ypred')
        
        score = -1 * mean_squared_error(y, ypred)
        return score 

    # Adapt fit in LinearModelCV
    # def fit_cv(self, X, y, *, lams=None, n_lams=100, cv=None,
    #            verbose=False, n_jobs=None, random_state=None):
        
    #     # # This makes sure that there is no duplication in memory.
    #     # # Dealing right with copy_X is important in the following:
    #     # # Multiple functions touch X and subsamples of X and can induce a
    #     # # lot of duplication of memory
    #     # copy_X = self.copy_X and self.fit_intercept

    #     # check_y_params = dict(copy=False, dtype=[np.float64, np.float32],
    #     #                       ensure_2d=False)
    #     # if isinstance(X, np.ndarray) or sparse.isspmatrix(X):
    #     #     # Keep a reference to X
    #     #     reference_to_old_X = X
    #     #     # Let us not impose fortran ordering so far: it is
    #     #     # not useful for the cross-validation loop and will be done
    #     #     # by the model fitting itself

    #     #     # Need to validate separately here.
    #     #     # We can't pass multi_ouput=True because that would allow y to be
    #     #     # csr. We also want to allow y to be 64 or 32 but check_X_y only
    #     #     # allows to convert for 64.
    #     #     check_X_params = dict(accept_sparse='csc',
    #     #                           dtype=[np.float64, np.float32], copy=False)
    #     #     X, y = self._validate_data(X, y,
    #     #                                validate_separately=(check_X_params,
    #     #                                                     check_y_params))
    #     #     if sparse.isspmatrix(X):
    #     #         if (hasattr(reference_to_old_X, "data") and
    #     #                 not np.may_share_memory(reference_to_old_X.data, X.data)):
    #     #             # X is a sparse matrix and has been copied
    #     #             copy_X = False
    #     #     elif not np.may_share_memory(reference_to_old_X, X):
    #     #         # X has been copied
    #     #         copy_X = False
    #     #     del reference_to_old_X
    #     # else:
    #     #     # Need to validate separately here.
    #     #     # We can't pass multi_ouput=True because that would allow y to be
    #     #     # csr. We also want to allow y to be 64 or 32 but check_X_y only
    #     #     # allows to convert for 64.
    #     #     check_X_params = dict(accept_sparse='csc',
    #     #                           dtype=[np.float64, np.float32], order='F',
    #     #                           copy=copy_X)
    #     #     X, y = self._validate_data(X, y,
    #     #                                validate_separately=(check_X_params,
    #     #                                                     check_y_params))
    #     #     copy_X = False

    #     if y.shape[0] == 0:
    #         raise ValueError("y has 0 samples: %r" % y)
        
    #     if X.shape[0] != y.shape[0]:
    #         raise ValueError("X and y have inconsistent dimensions (%d != %d)"
    #                          % (X.shape[0], y.shape[0]))
        
    

    # Used as path in AdditiveModelCV
    # Adapted from enet_path
    # Make alpha required for now 
    # def __call__(self, X, y, *, alphas, l1_ratio=1, 
    #              eps=1e-3, copy_X=True, fhat_init=None, verbose=False,
    #              return_n_iter=False, check_input=True, **params):
        
    #     if check_input:
    #         X = check_array(X, accept_sparse='csc', 
    #                         dtype=[np.float64, np.float32], 
    #                         order='F', copy=copy_X)
    #         y = check_array(y, accept_sparse='csc', dtype=X.dtype.type,
    #                         order='F', copy=False, ensure_2d=False)
        
    #     n_samples, n_features = X.shape
        
    #     multi_output = False
    #     if y.ndim != 1:
    #         multi_output = True
    #         _, n_outputs = y.shape
        
    #     alphas = np.sort(alphas)[::-1]
    #     n_alphas = len(alphas)
        
    #     # # Probably unnecessary
    #     # tol = params.get('tol', 1e-4)
    #     # max_iter = params.get('max_iter', 1000)
    #     # n_iters = []
        
    #     rng = check_random_state(params.get('random_state', None))
        
    #     if not multi_output:
    #         # TODO: Need to change n_features to length of fhat + 1
    #         coefs = np.empty((n_features, n_alphas), dtype=X.dtype)
                
            
# def gsam_path(X, y, *, n_alphas=10, alphas=None, l1_ratio=1, eps=1e-3,
#               copy_X=True, coef_init=None, verbose=False, 
#               return_n_iter=False, check_input=True, **params):
                
# class SparseAdditiveRegressionCV(RegressorMixin, AdditiveModelCV):
#     pass 
        
