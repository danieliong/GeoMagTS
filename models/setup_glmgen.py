from distutils.core import setup 
from distutils.extension import Extension
from Cython.Build import cythonize 
import numpy as np 
 
glmgen_ext = Extension(
    name='glmgen',
    sources=['glmgen.pyx'],
    include_dirs=['glmgen_c/include', np.get_include()],
    libraries=['glmgen'],
    library_dirs=['glmgen_c/lib', 'glmgen_c/obj'],
    runtime_library_dirs=['glmgen_c/lib', 'glmgen_c/obj']
)

setup(
    name='glmgen',
    ext_modules=cythonize([glmgen_ext]),
    include_dirs=[np.get_include()]
)
