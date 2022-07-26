# Authors: Hugo Richard, Pierre Ablin
# License: BSD 3 clause

from ._multiviewica import multiviewica, _multiview_ica_main
from ._groupica import groupica
from ._permica import permica, _hungarian
from .optimization_tau import _optimization_tau, _create_sources, _loss_delay

__version__ = '0.1'
