import pytest
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('../..')
from GeoMagTS.glmgen import trendfilter
from GeoMagTS.models.additive import TF_DEFAULT_KWARGS, _admm_tf, SparseAdditiveRegression

def create_gsam_data(n, p, sigma):

    X = np.random.randn(size=(n, p))

    f_mat = np.zeros((n, p))
    f_mat[:, 0] = np.sin((3*np.pi) / (X[:, 0]+0.5))
    f_mat[:, 1] = (X[:, 1] * (X[:, 1] - (1/3)))
    f_mat = f_mat - f_mat.mean(axis=0)

    y = 5 + f_mat.sum(axis=-1) + (sigma*np.random.randn(n))
    return X, y, f_mat

def f_0(x): return - (4 * (x**3)) + (0.5 * (x**2)) + 2
def f_1(x): return np.sin((3*np.pi) / (x+0.5))
def f_2(x): return -2 * x

def create_ar_data(n, p, *, sigma=0.01, intercept=1, **kwargs):
    
    y = np.zeros(n)
    X = np.random.randn(n,p)
    
    f_mat = np.zeros((n, 2*p))
    
    for i in range(2,n):
        f_mat[i,0] = 0.8 * y[i-1]
        f_mat[i,1] = -.5 * y[i-2]
        f_mat[i,2] = f_0(X[(i-1),0])
        f_mat[i,3] = f_1(X[(i-2),0])
        f_mat[i,4] = f_2(X[(i-3), 0])
        
        y[i] = np.sum(f_mat[i,:]) + (sigma*np.random.randn(1))
    norm_y = np.sqrt(np.mean(y**2))
    y = (y - y.mean()) / norm_y
    
    return X, y, f_mat

X, y, f_mat = create_ar_data(1000, 5)

gsam = SparseAdditiveRegression(lam=0.1, verbose=True)
gsam.fit(X, y)

ypred = gsam.predict(X)
fhat = gsam.predict(X, output='fhat')

# plt.plot(y, color='black')
# plt.plot(ypred, color='red')

# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].scatter(X[:, 0], f_mat[:, 0], color='black', s=1)
# ax[0].scatter(X[:, 0], gsam.fhat_[:, 0], color='red', s=1)
# ax[1].scatter(X[:, 1], f_mat[:, 1], color='black', s=1)
# ax[1].scatter(X[:, 1], gsam.fhat_[:, 1], color='red', s=1)
# plt.show()
    
        
    
