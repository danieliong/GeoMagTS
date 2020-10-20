import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from glmgen import trendfilter


def t_loss(y, f_hat, intercept, df=10):
    # TODO: Need to figure out how to estimate scale parameter.
    pass

def squared_loss(y, fhat, intercept):
    y_fitted = np.sum(fhat, axis=1) + intercept 
    return np.mean((y - y_fitted)**2)

def gaussian_grad(y, fhat, intercept):
    y_fitted = np.sum(fhat, axis=1) + intercept
    n = y.shape[0]
    return -(1/n) * (y - y_fitted) 
 
 
TF_DEFAULT_KWARGS = dict(
    family=0,
    max_iter=200, 
    lam_flag=0,
    nlambda=1, 
    lambda_min_ratio=1e-5, 
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
 
class SparseAdditiveRegression(BaseEstimator,  RegressorMixin):
    def __init__(self, 
                 loss='gaussian',
                 method='tf',
                 order=2,
                 init_fhat=None,
                 init_intercept=None,
                 lam = 1,
                 init_step_size=None,
                 max_iter=100,
                 tol=1e-4,
                 alpha=0.5, # For line search
                 df=10,
                 **kwargs):
        self.loss = loss
        self.method = method
        self.order = order
        self.init_fhat = init_fhat
        self.init_intercept = init_intercept
        self.lam = lam
        self.init_step_size = init_step_size
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.df = df
        
        self.tf_kwargs = dict(TF_DEFAULT_KWARGS)
        self.tf_kwargs.update(**kwargs)
    
    # TODO: Validation for loss, method
        
    @property
    def lam1(self):
        return self.lam
    
    @property 
    def lam2(self):
        return self.lam**2
    
    # REMOVE. This is for trendfilter. self.loss is not the same as family 
    @property
    def family_(self):
        if self.loss == 'gaussian':
            return 0
        else:
            raise ValueError(
                self.loss + "loss function is not supported yet.")
    
    def _prox_grad_update_tf(self, x_ord, y_ord, lam1, lam2):
        n = y_ord.shape[0]
        
        tf = trendfilter(x_ord=x_ord, y_ord=y_ord, w_ord=np.ones(n),
                         k=self.order, 
                         lam=np.array([n*lam2]).astype(np.float), 
                         **self.tf_kwargs)
        f_hat = tf['f_hat'][0,:]
        x_thinned = tf['x']
        # n_thinned = x_thinned.shape[0]
        
        if x_thinned.shape[0] != n:
            f_hat = np.interp(x_ord, xp=x_thinned, fp=f_hat)
        
        f_hat_norm = np.sqrt(np.mean(f_hat**2))
        f_thres = max(1 - (lam1 / f_hat_norm), 0) * f_hat
        
        return f_thres
    
    # GetZ
    # fhat: n*p matrix
    def _prox_grad_fixed_step(self, X, y, step, f_hat, intercept):
        # Order X, y here 
        # Need C function for tf here 
        n, p = f_hat.shape
        
        # Order X
        ord = np.argsort(X, axis=0)
        X_ord = np.sort(X, axis=0)
        
        if self.loss == 'gaussian':
            grad = gaussian_grad(y, f_hat, intercept).reshape(-1,1)
        
        intercept_new = intercept - (step * sum(grad))
        inter_step = f_hat - step * grad
        
        f_new = np.zeros((n,p))
        for i in range(p):
            inter_step_i = inter_step[ord[:,i], i]
            
            if self.method == 'tf':
                f_new[ord[:,i], i] = self._prox_grad_update_tf(
                    x_ord=X_ord[:,i], 
                    y_ord=inter_step_i-np.mean(inter_step_i), 
                    lam1=(self.lam1*step)/n,
                    lam2=(self.lam2*step)/n,
                    )
        
        return intercept_new, f_new    
    
    # LineSearch
    def _prox_grad_line_search(self, X, y, f_hat, intercept):
        if self.loss == 'gaussian':
            loss = squared_loss(y, f_hat, intercept)
            grad = gaussian_grad(y, f_hat, intercept)
        
        converged = False
        step = self.init_step_size
        
        while not converged:
            intercept_new, f_new = self._prox_grad_fixed_step(
                X, y, step, f_hat, intercept)
            
            lhs = squared_loss(y, f_new, intercept_new)
            
            rhs = loss + np.sum(grad) * (intercept_new - intercept) + np.sum(np.dot(grad, (f_new - f_hat))) +  (1/(2*step)) * (np.sum((f_new - f_hat)**2) + (intercept_new - intercept)**2)
            
            if lhs <= rhs:
                converged = True
            else:
                step = self.alpha * step
        
        return intercept_new, f_new
        
    
    # proxGrad_one 
    def fit(self, X, y=None):
        # TODO: Input validation
        
        y_norm = np.sqrt(np.mean(y**2))
        if self.lam >= y_norm:
            raise ValueError("self.lam >= ||y||_n. This will result in f_hat being all zeros.")
        
        n, p = X.shape
        
        if self.init_fhat is None:
            self.init_fhat = np.zeros(X.shape)
        
        if self.init_intercept is None:
            self.init_intercept = np.mean(y) 
        
        if self.init_step_size is None:
            self.init_step_size = y.shape[0]
        
        # proxGrad_one
        iter = 1
        converged = False
        
        intercept_old, f_old = self.init_intercept, self.init_fhat
        
        while not converged and iter < self.max_iter:
            intercept_new, f_new = self._prox_grad_line_search(
                X, y, f_old, intercept_old)
            
            numer = np.mean((f_new - f_old)**2 + (intercept_new - intercept_old)**2)
            denom = np.mean(f_old**2) + (intercept_old**2)
            rel_change = numer / (denom+1e-30)
            
            if (rel_change <= self.tol):
                converged = True
            else:
                iter = iter + 1
                intercept_old, f_old = intercept_new, f_new
        
        self.intercept_ = intercept_new
        self.fhat_ = f_new
        
        return self
        
        
    def predict(self, X, y=None):
        pass
