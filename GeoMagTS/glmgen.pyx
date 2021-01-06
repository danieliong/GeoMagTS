from __future__ import print_function
import cython
import numpy as np 
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport array
import array
from libc.stdlib cimport free 
from libc.string cimport memcpy 
from libc.stdio cimport printf
np.import_array()

# Import data structures from glmgen C library 
cdef extern from "cs.h":
    ctypedef  ptrdiff_t csi  
    ctypedef struct cs: 
        csi nzmax    
        csi m         
        csi n         
        csi *p         
        csi *i        
        double *x     
        csi nz

    ctypedef struct css:
        csi *pinv
        csi *q
        csi *parent
        csi *cp
        csi *leftmost
        csi m2
        double lnz
        double unz

    ctypedef struct csn:
        cs *L
        cs *U
        csi *pinv
        double *B
    
    ctypedef struct csd:
        csi *p
        csi *q
        csi *r
        csi *s
        csi nb
        csi rr [5]
        csi cc [5]
    
    cs *cs_transpose(const cs *A, csi values)
 
cdef extern from "utils.h":
    ctypedef struct gqr:
        csi m
        csi n
        css *S
        csn *N
        double * W

    ctypedef double (*func_RtoR)(double)

    # Thinning function for ill-conditioned and large data matrices
    void thin(double * x, double * y, double * w,
	int n, int k, double ** xt, double ** yt,
	double ** wt, int * nt_ptr, double tol)
    
    double glmgen_factorial(int n)

    gqr * glmgen_qr(const cs *A)


# Import fitting functions from glmgen C library
cdef extern from "tf.h":
    double * tf_admm_default(double * y, int n)

    void tf_admm(double * x, double * y, double * w, int n, int k, int family,
                int max_iter, int lam_flag, double * lam,
                int nlambda, double lambda_min_ratio, int * df,
                double * beta, double * obj, int * iter, int * status,
                double rho, double obj_tol, double obj_tol_newton, double alpha_ls,
                double gamma_ls, int max_iter_ls, int max_inner_iter, int verbose)

    void tf_admm_gauss(double * x, double * y, double * w, int n, int k,
                    int max_iter, double lam, int * df,
                    double * beta, double * alpha, double * u,
                    double * obj, int * iter,
                    double rho, double obj_tol, cs * DktDk, int verbose)

    void tf_admm_glm(double * x, double * y, double * w, int n, int k,
                    int max_iter, double lam, int * df,
                    double * beta, double * alpha, double * u,
                    double * obj, int * iter,
                    double rho, double obj_tol, double obj_tol_newton, double alpha_ls,
                    double gamma_ls, int max_iter_ls, int max_iter_newton,
                    cs * DktDk, func_RtoR b, func_RtoR b1, func_RtoR b2, int verbose)

    void tf_dp(int n, double * y, double lam, double * beta)
    void tf_dp_weight(int n, double * y, double * w, double lam, double * beta)

    void tf_predict(double * x, double * beta,  int n, int k, int family,
                    double * x0, int n0, double * pred, double zero_tol)
    void tf_predict_gauss(double * x, double * beta, int n, int k,
                        double * x0, int n0, double * pred, double zero_tol)
    void poly_coefs(double * x, double * beta, int k, double * phi)

    double tf_maxlam(int n, double * y, gqr * Dt_qr, double * w)

    cs * tf_calc_dk(int n, int k, const double * x)
    cs * tf_calc_dktil(int n, int k, const double * x)
    void tf_dx(double * x, int n, int k, double * a, double * b)
    void tf_dtx(double * x, int n, int k, double * a, double * b)
    void tf_dxtil(double * x, int n, int k, double * a, double * b)
    void tf_dtxtil(double * x, int n, int k, double * a, double * b)

    double tf_obj(double * x, double * y, double * w, int n, int k, double
    lam, int family, double * beta, double * buf)
    double tf_obj_gauss(double * x, double * y, double * w, int n, int k, double
    lam, double * beta, double * buf)
    double tf_obj_glm(double * x, double * y, double * w, int n, int k, double
    lam, func_RtoR b, double * beta, double * buf)

def maxlam_tf(np.ndarray x_ord, np.ndarray y_ord, np.ndarray w_ord, int k):

    cdef int n = y_ord.shape[0]

    cdef double[:] x_view = np.ascontiguousarray(x_ord, dtype=np.float64)
    cdef double[:] y_view = np.ascontiguousarray(y_ord, dtype=np.float64)
    cdef double[:] w_view = np.ascontiguousarray(w_ord, dtype=np.float64)

    cdef double * x_ptr = &x_view[0]
    cdef double * y_ptr = &y_view[0]
    cdef double * w_ptr = &w_view[0]

    cdef cs * D = tf_calc_dk(n, k+1, x_ptr)
    cdef cs * Dt = cs_transpose(D, 1)
    cdef gqr * Dt_qr = glmgen_qr(Dt)

    cdef double max_lam = tf_maxlam(n, y_ptr, Dt_qr, w_ptr)

    return max_lam

# Cython wrapper for tf_admm function in the glmgen C library
@cython.boundscheck(False)
@cython.wraparound(False)
def trendfilter(
    np.ndarray[double, ndim=1, mode="c"] x_ord, 
    np.ndarray[double, ndim=1, mode="c"] y_ord, 
    np.ndarray[double, ndim=1, mode="c"] w_ord, int k, 
    int family, int max_iter, int lam_flag, 
    np.ndarray[double, ndim=1, mode="c"] lam, int nlambda, 
    double lambda_min_ratio, double rho, double obj_tol, 
    double obj_tol_newton, double alpha_ls, double gamma_ls, 
    int max_iter_ls, int max_iter_newton, int thinning, double x_tol, 
    int verbose, **kwargs):

    cdef int n = y_ord.shape[0]

    cdef double * x_ptr = NULL
    cdef double * y_ptr = NULL
    cdef double * w_ptr = NULL

    cdef double * xt_ptr = NULL
    cdef double * yt_ptr = NULL
    cdef double * wt_ptr = NULL
    cdef int nt

    cdef array.array int_temp = array.array('i', [])
    cdef array.array d_temp = array.array('d', [])
    cdef array.array df
    cdef array.array fhat
    cdef array.array obj
    cdef array.array iter
    cdef array.array status

    cdef double[::1] xt

    # Thinning
    if thinning == 1:
        thin(&x_ord[0], &y_ord[0], &w_ord[0], n, k, &xt_ptr, &yt_ptr, 
        &wt_ptr, &nt, x_tol)

    # If data was not thinned
    if xt_ptr == NULL:
        x_ptr = &x_ord[0]
        y_ptr = &y_ord[0]
        w_ptr = &w_ord[0]
    else:
        n = nt
        x_ptr = xt_ptr
        y_ptr = yt_ptr
        w_ptr = wt_ptr

    df = array.clone(int_temp, nlambda, zero=False)
    fhat = array.clone(d_temp, n*nlambda, zero=False)
    obj = array.clone(d_temp, nlambda*max_iter, zero=True)
    iter = array.clone(int_temp, nlambda, zero=False)
    status = array.clone(int_temp, nlambda, zero=False)

    tf_admm(x_ptr, y_ptr, w_ptr, n, k, family, max_iter, lam_flag, &lam[0],
    nlambda, lambda_min_ratio, &df.data.as_ints[0], &fhat.data.as_doubles[0], 
    &obj.data.as_doubles[0], &iter.data.as_ints[0], &status.data.as_ints[0], rho, obj_tol, obj_tol_newton, alpha_ls, gamma_ls, max_iter_ls, max_iter_newton, verbose)


    if xt_ptr == NULL:
        res = dict(df=np.asarray(df, dtype=np.int16), 
        f_hat=np.asarray(fhat, dtype=np.float64).reshape((nlambda, n)), 
        obj=np.asarray(obj, dtype=np.float64).reshape((nlambda, max_iter)), 
        iter=np.asarray(iter, dtype=np.int16), 
        status=np.asarray(status, dtype=np.int16), 
        x=None)
    else:
        free(yt_ptr)
        free(wt_ptr)

        xt = <double[:nt]> xt_ptr
        free(xt_ptr)
        res = dict(df=np.asarray(df, dtype=np.int16), 
        f_hat=np.asarray(fhat, dtype=np.float64).reshape((nlambda, n)), 
        obj=np.asarray(obj, dtype=np.float64).reshape((nlambda, max_iter)), 
        iter=np.asarray(iter, dtype=np.int16), 
        status=np.asarray(status, dtype=np.int16), 
        x=xt)

    return res


def predict_tf(
    np.ndarray x0, np.ndarray x, np.ndarray beta, int k, int family, 
    double zero_tol, **kwargs):

    cdef int n = x.shape[0]
    cdef int n0 = x0.shape[0]

    cdef double[:] x_view = np.ascontiguousarray(x, dtype=np.float64)
    cdef double[:] x0_view = np.ascontiguousarray(x0, dtype=np.float64)
    cdef double[:] beta_view = np.ascontiguousarray(beta, dtype=np.float64)
    cdef double * x_ptr = &x_view[0]
    cdef double * x0_ptr = &x0_view[0]
    cdef double * beta_ptr = &beta_view[0]

    # Allocate space for predictions
    pred_c = <double*> PyMem_Malloc(n0 * sizeof(double))

    tf_predict(x_ptr, beta_ptr, n, k, family, x0_ptr, n0, pred_c, zero_tol)

    cdef np.ndarray pred = np.asarray(<double[:n0]> pred_c)
    return pred

def poly_coefs_tf(np.ndarray x, int k, np.ndarray beta):

    cdef double[:] x_view = np.ascontiguousarray(x, dtype=np.float64)
    cdef double[:] beta_view = np.ascontiguousarray(beta, dtype=np.float64)
    cdef double * x_ptr = &x_view[0]
    cdef double * beta_ptr = &beta_view[0]

    # Allocate space for output 
    coefs_c = <double*> PyMem_Malloc((k+1)*sizeof(double))

    poly_coefs(x_ptr, beta_ptr, k, coefs_c)

    cdef np.ndarray coefs = np.asarray(<double[:k]> coefs_c)
    return coefs

def falling_fact_coefs_tf(np.ndarray x, int k, np.ndarray beta):
    
    cdef int n = x.shape[0]
    cdef double k_fac
    cdef double[:] x_view = np.ascontiguousarray(x, dtype=np.float64)
    cdef double[:] beta_view = np.ascontiguousarray(beta, dtype=np.float64)
    cdef double * x_ptr = &x_view[0]
    cdef double * beta_ptr = &beta_view[0]

    coefs_c = <double*> PyMem_Malloc((n-k-1)*sizeof(double))

    tf_dx(x_ptr, n, (k+1), beta_ptr, coefs_c)
    k_fact = glmgen_factorial(k)

    cdef np.ndarray coefs = np.asarray(<double[:(n-k-1)]> coefs_c)
    coefs = coefs / k_fact

    return coefs



def multiplyD_tf(
    np.ndarray[double, ndim=1, mode="c"] x, 
    np.ndarray[double, ndim=1, mode="c"] a
    int k):

    cdef int n = x.shape[0]

    cdef array.array d_temp = array.array('d', [])
    cdef array.array b 

    b = array.clone(d_temp, n, zero=False)

    tf_dx(&x[0], n, k, &a[0], &b.data.as_doubles[0])

    # cdef np.ndarray b = np.asarray(<double[:n]> b_c)
    return np.asarray(b, dtype=np.float64)







