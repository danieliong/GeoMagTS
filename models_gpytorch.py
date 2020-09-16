import torch
import gpytorch
from gpytorch.lazy import RootLazyTensor, MatmulLazyTensor, MulLazyTensor, lazify

'''
GPyTorch Means 
'''

class PersistenceMean(gpytorch.means.Mean):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x[:,0]

'''
GPyTorch Kernels 
'''

class MLPKernel(gpytorch.kernels.Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter(
            name="raw_w", 
            parameter=torch.nn.Parameter(torch.zeros(1))
            )
        self.register_parameter(
            name="raw_b",
            parameter=torch.nn.Parameter(torch.zeros(1))
        )
        
        self.positive_constraint = gpytorch.constraints.Positive()
        self.register_constraint("raw_w", self.positive_constraint)
        self.register_constraint("raw_b", self.positive_constraint)
    
    @property
    def w(self):
        return self.positive_constraint.transform(self.raw_w)
    
    @property
    def b(self):
        return self.positive_constraint.transform(self.raw_b)
    
    @w.setter
    def w(self, value):
        return self._set_w(value)
    
    @b.setter
    def b(self, value):
        return self._set_b(value)
    
    def _set_w(self, w):
        if not torch.is_tensor(w):
            w = torch.as_tensor(w).to(self.raw_w)
            
        set.initialize(raw_w=self.positive_constraint.inverse_transform(w))
        
    def _set_b(self, b):
        if not torch.is_tensor(b):
            b = torch.as_tensor(b).to(self.raw_b)
            
        set.initialize(raw_b=self.positive_constraint.inverse_transform(b))
    
    def forward(self, x, y, **params):
        one = torch.Tensor([1])
        
        if x.dim() == 2 and y.dim() == 2:
            numer = torch.addmm(self.b, x, y.transpose(-2,-1))
        else:
            numer = (self.w * x.matmul(y.transpose(-2, -1))) + self.b
        
        norm_sq_x = x.matmul(x.transpose(-2,-1)).diag()
        if x.size() == y.size() and torch.equal(x, y):
            denom = (self.w * norm_sq_x.ger(norm_sq_x)) + self.b + one
        else:
            norm_sq_y = y.matmul(y.transpose(-2,-1)).diag()
            denom = torch.ger(
                (self.w*norm_sq_x)+self.b+one,
                (self.w*norm_sq_y)+self.b+one
            ).sqrt()
        
        res = torch.asin(numer.div(denom))
        return res 
        
    # def forward(self, x, y, **params):
    #     # Convert to matrix so that it can be added to matrices 
    #     ones_ = lazify(torch.ones(*self.batch_shape, x.shape[0], x.shape[0]))
    #     b_ = lazify(self.b * ones_)
    #     xTx = RootLazyTensor(x)
        
    #     if x.size() == y.size() and torch.equal(x, y):
    #         numer = (self.w * xTx) + b_
    #         denom = 
    #         # denom = numer + ones_
    #         res = torch.asin(numer.evaluate().div(denom.evaluate().abs()))
    #     else:
    #         yTy = RootLazyTensor(y)
    #         xTy = MatmulLazyTensor(x, y.transpose(-2,-1))
    #         numer = (self.w * xTy) + b_
    #         denom_sq = MulLazyTensor(
    #             (self.w*xTx)+ b_+ones_,
    #             (self.w*yTy)+b_+ones_
    #             )
    #         res = torch.asin(
    #             numer.evaluate().div(
    #                 denom_sq.evaluate().sqrt()
    #                 )
    #             )
    #         # NOTE: cant take element-wise square root of LazyTensor
    #         # If have time, create LazyTensor that can take element-wise square
    #         # root
    #     return res
        
class StudentTKernel(gpytorch.kernels.Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter(
            name="raw_d",
            parameter=torch.nn.Parameter(torch.Tensor([0]))
            )
        self.positive_constraint = gpytorch.constraints.Positive()
        self.register_constraint("raw_d", self.positive_constraint)
        
    @property 
    def d(self):
        return self.positive_constraint.transform(self.raw_d)
    
    @d.setter
    def d(self, value):
        return self._set_d(value)
    
    def _set_d(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_d)
        self.initialize(raw_d=self.positive_constraint.inverse_transform(value))
    
    def forward(self, x, y, **params):
        dist = self.covar_dist(x, y, **params)
        denom = torch.Tensor([1]).add(dist.pow(self.d))
        res = torch.Tensor([1]).div(denom)
        return res
    
'''
GPyTorch Models
'''

class SimpleGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class GeoMagExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = PersistenceMean()
        self.covar_module = MLPKernel() + StudentTKernel()
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
