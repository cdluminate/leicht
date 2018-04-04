from distutils.core import setup, Extension

module_leicht = Extension('leicht',
                          sources=['leicht.i'],
                          swig_opts=['-c++'],
                          include_dirs=['/usr/include/hdf5/serial'],
                          libraries=['jsoncpp', "hdf5_cpp", "hdf5_hl_cpp",
                            "hdf5_serial", "blas"],
                         )
setup (name = 'leicht',
       ext_modules = [module_leicht],
       py_modules = ["leicht"],
       )
