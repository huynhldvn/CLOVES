"""
CLOVES: Cleft-Overstep Optimization for Deep Neural Networks & LLMs
"""

from ._version import __version__
from .bisector import CLOVES, CLOVES_Bisector
from .perpendicular import CLOVES_Perpendicular
from .affine import CLOVES_Affine
from .adaptive import CLOVES_Adaptive
from .llm import CLOVES_LLM
from .apex import CLOVES_Apex_Asymmetric
from .matrix_polar import CLOVES_Matrix_Polar

__all__ = [
    "__version__",
    "CLOVES",
    "CLOVES_Bisector",
    "CLOVES_Perpendicular",
    "CLOVES_Affine",
    "CLOVES_Adaptive",
    "CLOVES_LLM",
    "CLOVES_Apex_Asymmetric",
    "CLOVES_Matrix_Polar",
]
