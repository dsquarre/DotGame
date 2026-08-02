from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("env.gameenv", ["env/gameenv.pyx"]),
    Extension("agents.minmax", ["agents/minmax.pyx"]),
    Extension("agents.mcts", ["agents/mcts.pyx"]),
    Extension("agents.tabular_q", ["agents/tabular_q.pyx"])
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    include_dirs=[np.get_include()]
)
