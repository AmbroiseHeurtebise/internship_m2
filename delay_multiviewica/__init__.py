# Authors: Hugo Richard, Pierre Ablin
# License: BSD 3 clause

from ._multiviewica import multiviewica, _multiview_ica_main
from ._multiviewica_test1 import multiviewica_test1
from ._multiviewica_test2 import multiviewica_test2
from ._multiviewica_test3 import multiviewica_test3
from ._multiviewica_test5 import multiviewica_test5
from ._groupica import groupica
from ._permica import permica, _hungarian
from .optimization_tau import _optimization_tau, _create_sources, _loss_delay, _apply_delay, _apply_delay_one_sub, create_sources_pierre, _loss_delay_ref, _optimization_tau_approach1, _optimization_tau_approach2
from ._univiewica import univiewica

__version__ = '0.1'
