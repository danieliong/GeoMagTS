import pytest
import os
import numpy as np

os.chdir('../..')
from GeoMagTS.glmgen import trendfilter 
from GeoMagTS.models.additive import TF_DEFAULT_KWARGS, _admm_tf

def create_tf_data(n, f=None, sigma=1, **kwargs):
    if f is None:
        f = lambda x: - 0.5 * (x**2) + 2
            
    x = np.random.normal(size=n)
    y = f(x, **kwargs) + (sigma*np.random.randn(n))
    
    order = np.argsort(x)
    return x[order], y[order]

x, y = create_tf_data(10000)
n = y.shape[0]
res = _admm_tf(x, (y-np.mean(y)), lam=1e-4, **TF_DEFAULT_KWARGS)
plt.scatter(x, y, s=1)
plt.plot(x, res[0])



def test_admm_tf():
    pass 
    
def test_trendfilter():
    pass 
