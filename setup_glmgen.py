from distutils.core import setup 
from distutils.extension import Extension
from Cython.Build import cythonize 
import numpy as np 
 
glmgen_ext = Extension(
    name='glmgen',
    sources=['glmgen.pyx'],
    libraries=['glmgen'],
    include_dirs=['./glmgen_c/include/', np.get_include()],
    library_dirs=['./glmgen_c/lib/'],
    runtime_library_dirs=['./glmgen_c/lib'],
    # extra_link_args=['-L./glmgen_c/lib/'],
)

setup(
    name='glmgen',
    ext_modules=cythonize([glmgen_ext]),
    include_dirs=[np.get_include()]
)
